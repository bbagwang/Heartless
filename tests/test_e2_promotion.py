"""E2: promotion statistics robust to dependence across days. The absolute path's t_day and the relative path's paired
daily-difference t are Newey-West (HAC) t statistics over the calendar days (Bartlett kernel, lag 5), and the
checkpoint thresholds are recalibrated (tools/calibrate_promotion.py) over the C2-B scenarios plus AR(1) day factors and
zero-mean block-switching edges. With lag 0 both statistics are exactly the C2-B ones. Synthetic records only."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from heartless.config import Settings
from heartless.execution import stats as S
from heartless.learning import research as R
from heartless.learning.research import judge_challenger
from heartless.util.timeutil import MS_DAY, MS_HOUR

ST = Settings(_env_file=None)
T0 = 1_780_000_000_000 // MS_DAY * MS_DAY  # a UTC midnight
ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "docs" / "results" / "e2_promotion_calibration.json"


def _cal():
    spec = importlib.util.spec_from_file_location("calibrate_promotion", ROOT / "tools" / "calibrate_promotion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_promotion"] = mod
    spec.loader.exec_module(mod)
    return mod


def _trades(days: list[list[float]], engine: str = "challenger-1", start: int = T0) -> list[dict]:
    out = []
    for d, rs in enumerate(days):
        for k, r in enumerate(rs):
            out.append({"position_id": f"{engine}-{d}-{k}", "engine": engine, "r_multiple": r, "pnl": r * 50.0,
                        "exit_time": start + d * MS_DAY + (k + 1) * MS_HOUR, "entry_time": start + d * MS_DAY,
                        "alpha": "squeeze_breakout"})
    return out


def _nw(x: np.ndarray, lag: int) -> float:
    """Textbook Newey-West variance sum of a demeaned series (independent of the production code)."""
    u = x - x.mean()
    v = float(u @ u)
    for lg in range(1, lag + 1):
        v += 2 * (1 - lg / (lag + 1)) * float(u[lg:] @ u[:-lg])
    return v


# --- the statistics -----------------------------------------------------------------------------------------------------
def test_with_lag_zero_the_hac_statistics_are_the_c2b_ones():
    rng = np.random.default_rng(1)
    for _ in range(20):
        days = [list(rng.normal(0.05, 1.0, rng.integers(0, 4))) for _ in range(40)]
        a, b = _trades(days), _trades([list(rng.normal(0, 1, 2)) for _ in range(40)], "paper")
        assert S.day_hac_t(a, 0) == pytest.approx(S.day_clustered_t(a), rel=1e-9, abs=1e-12)
        pd0 = S.paired_daily_diff(a, b, T0, T0 + 40 * MS_DAY - 1)
        pd5 = S.paired_daily_diff(a, b, T0, T0 + 40 * MS_DAY - 1, lag=5)
        assert pd0["t"] == pd0["t_iid"] == pd5["t_iid"] and pd5["lag"] == 5
        diffs = np.array([sum(t["r_multiple"] for t in a if S.exit_day(t) == T0 // MS_DAY + i)
                          - sum(t["r_multiple"] for t in b if S.exit_day(t) == T0 // MS_DAY + i) for i in range(40)])
        k = len(diffs)
        for lag in (0, 1, 5):
            want = diffs.mean() * k / math.sqrt(_nw(diffs, lag) * k / (k - 1))
            assert S.hac_t(list(diffs), lag) == pytest.approx(want, rel=1e-9)
        assert pd5["t"] == pytest.approx(S.hac_t(list(diffs), 5), rel=1e-12)
    assert S.bartlett_weights(5) == pytest.approx([5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
    assert S.hac_t([0.3] * 10) == 0.0 and S.hac_t([1.0]) == 0.0  # no dispersion / one day: no t


def test_the_hac_t_discounts_runs_of_good_and_bad_days():
    # the same daily outcomes, once in runs (positively autocorrelated) and once interleaved
    good, bad = [[0.9, -0.2]] * 1, [[-0.6, 0.1]]
    runs = (good * 6 + bad * 4) * 6  # 60 days: 6-day good runs, 4-day bad runs
    mixed = (good * 3 + bad * 2) * 12
    t_runs, t_mixed = S.r_stats(_trades(runs)), S.r_stats(_trades(mixed))
    assert t_runs["t_day"] == pytest.approx(t_mixed["t_day"])  # the day-clustered t ignores the order of the days
    assert t_runs["t_hac"] < 0.9 * t_runs["t_day"]  # runs: fewer independent days than days
    assert t_mixed["t_hac"] > t_mixed["t_day"]  # interleaved good / bad days are better evidence than independent ones
    assert S.r_stats(_trades(runs), lag=0)["t_hac"] == pytest.approx(t_runs["t_day"])


def test_the_vectorised_statistics_equal_the_production_ones():
    cal = _cal()
    assert cal.LAG == S.HAC_LAG == R.PROMOTION_HAC_LAG == 5 and cal.VAR_FLOOR == S.VAR_FLOOR
    rng = np.random.default_rng(3)
    for sc in (cal.Scenario("E", 2, 0.30, 0.3, 1.0, phi=0.6), cal.Scenario("T", 3, 0.6, 0.0, block=10, amp=0.1),
               cal.Scenario("abs", 1, 0.45, 0.3, phi=0.3), cal.Scenario("I", 2, 0.45, 0.0, block=20, amp=0.1)):
        m = 10
        ch_tr, champ_tr = cal.simulate(sc, m, rng)
        ch, champ = cal.Record(ch_tr, m), (cal.Record(champ_tr, m) if champ_tr else None)
        pair = cal.Paired(ch, champ) if champ else None
        for i in range(m):
            def trades(tr):
                p, dd, r = tr
                return [{"r_multiple": float(r[j]), "exit_time": T0 + int(dd[j]) * MS_DAY + k + 1, "entry_time": T0}
                        for k, j in enumerate(np.where(p == i)[0])]
            a_all, b_all = trades(ch_tr), (trades(champ_tr) if champ_tr else [])
            for c in cal.CHECKPOINTS + (3, 7):
                cut = T0 + c * MS_DAY
                a, d = [t for t in a_all if t["exit_time"] < cut], c - 1
                st = S.r_stats(a)
                assert ch.t_hac(d)[i] == pytest.approx(st["t_hac"], abs=1e-6)
                assert ch.t_day(d)[i] == pytest.approx(st["t_day"], abs=1e-6)
                if champ is not None:
                    b = [t for t in b_all if t["exit_time"] < cut]
                    pd_ = S.paired_daily_diff(a, b, T0, cut - 1, lag=R.PROMOTION_HAC_LAG)
                    assert pair.t_hac(d)[i] == pytest.approx(pd_["t"], abs=1e-6)
                    assert pair.t(d)[i] == pytest.approx(pd_["t_iid"], abs=1e-6)


def test_the_dependence_scenarios_have_the_intended_structure():
    cal = _cal()
    rng = np.random.default_rng(5)
    z = cal._day_factor(rng, 4000, 0.6)
    assert np.corrcoef(z[:, 1:].ravel(), z[:, :-1].ravel())[0, 1] == pytest.approx(0.6, abs=0.02)
    assert z.std() == pytest.approx(1.0, abs=0.02)  # stationary unit variance
    e = cal._edge_days(rng, 2000, cal.Scenario("abs", 1, 0.45, 0.0, block=10, amp=0.1), 0.0)
    assert set(np.unique(e)) == {-0.1, 0.1} and abs(e.mean()) < 0.005  # zero-mean +-0.10R
    runs = np.diff(e, axis=1) != 0
    assert runs.sum(axis=1).max() <= 9  # at most one sign change per 10-day block
    grid = cal.scenarios()
    groups = {g: sum(1 for s in grid if s.group == g) for g in ("base", "ar", "block", "stress")}
    assert groups == {"base": 120, "ar": 120, "block": 240, "stress": 120}


# --- the calibrated thresholds ----------------------------------------------------------------------------------------
def test_the_thresholds_are_the_calibrated_ones():
    doc = json.loads(CALIBRATION.read_text())
    ch = doc["chosen"]
    assert {int(k): v for k, v in ch["normal"]["abs"].items()} == R.ABS_T_AT
    assert {int(k): v for k, v in ch["normal"]["rel"].items()} == R.REL_T_AT
    assert {int(k): v for k, v in ch["failed"]["abs"].items()} == R.FAILED_ABS_T_AT
    assert {int(k): v for k, v in ch["failed"]["rel"].items()} == R.FAILED_REL_T_AT
    assert tuple(R.ABS_T_AT) == R.PROMOTION_CHECKPOINTS and tuple(R.FAILED_ABS_T_AT) == R.FAILED_CHECKPOINTS
    for t in (R.ABS_T_AT, R.REL_T_AT, R.FAILED_ABS_T_AT, R.FAILED_REL_T_AT):
        v = list(t.values())
        assert v == sorted(v, reverse=True)  # strictest first: most of the error budget where the evidence is
    model = doc["model"]
    assert model["statistic"] == {"kind": "newey-west", "kernel": "bartlett", "lag_days": R.PROMOTION_HAC_LAG}
    assert model["dd_scaling"] is True and model["targets"] == {"normal": 0.05, "failed": 0.02}
    assert model["targeted_groups"] == ["base", "ar", "block"] and model["ar_phis"] == [0.3, 0.6]
    assert [b for b, _ in model["blocks"]] == [10, 20] and model["paths"] >= 20000
    for cls, target in (("normal", 0.05), ("failed", 0.02)):
        summ = doc["designs"][str(doc["chosen_delta"])][cls]["summary"]
        # an independent run (fresh seeds), Monte Carlo error included, in every targeted scenario
        assert summ["false_promotion"]["max"] <= target + 0.003
        for g in ("base", "ar", "block"):
            assert summ["false_promotion_" + g]["max"] <= target + 0.003, (cls, g)
        # the C2-B design on the same paths: it broke its target under dependence across days
        old = doc["c2b"][cls]["summary"]
        assert old["false_promotion_ar"]["max"] > target + 0.01 > summ["false_promotion_ar"]["max"]


# --- the production judge -----------------------------------------------------------------------------------------------
def _judge(days, champ=(), at_day=None, new_alphas=("discovered",)):
    at = T0 + int((at_day if at_day is not None else len(days)) * MS_DAY) + MS_HOUR
    return judge_challenger(_trades(days), _trades(list(champ), "paper"), ST, T0, at, list(new_alphas))


RUNS = ([[1.0, 0.4]] * 7 + [[-0.7, 0.1]] * 5) * 5 + [[1.0, 0.4]] * 2  # 62 days, 7-day good and 5-day bad runs


def test_the_absolute_path_decides_on_the_hac_t_day():
    d = _judge(RUNS, at_day=60.05)
    rec = S.r_stats(_trades(RUNS[:60]))
    assert d["path"] == "absolute" and d["checkpoint"] == 60 and d["challenger"]["t_hac"] == round(rec["t_hac"], 4)
    assert rec["t_day"] >= R.ABS_T_AT[60] > rec["t_hac"]  # independent days would have promoted it
    assert d["checks"][f"HAC t_day >= {R.ABS_T_AT[60]:g}"] is False and d["verdict"] == "hold"
    assert d["bar"]["min_t_hac"] == d["bar"]["min_t_day"] == R.ABS_T_AT[60] and d["bar"]["hac_lag"] == 5
    assert d["stat"] == {"kind": "newey-west", "kernel": "bartlett", "lag_days": 5}
    text = R.describe_decision(d)
    assert f"HAC(5일) t_day≥{R.ABS_T_AT[60]:g}" in text and f"HAC(5일) t_day {rec['t_hac']:+.2f}" in text


def test_the_relative_path_decides_on_the_hac_paired_t():
    champ = [[0.3, -0.3]] * 62
    d = _judge([[a + 0.3, b - 0.3] for a, b in RUNS], champ, at_day=60.05)
    cut = T0 + 60 * MS_DAY
    pd_ = S.paired_daily_diff([t for t in _trades([[a + 0.3, b - 0.3] for a, b in RUNS]) if t["exit_time"] < cut],
                              [t for t in _trades(champ, "paper") if t["exit_time"] < cut], T0, cut - 1, lag=5)
    assert d["path"] == "relative" and d["paired"]["t"] == round(pd_["t"], 4)
    assert d["paired"]["t_iid"] == round(pd_["t_iid"], 4) and pd_["t_iid"] > pd_["t"]
    assert d["checks"][f"paired t >= {R.REL_T_AT[60]:g}"] is (pd_["t"] >= R.REL_T_AT[60])
    assert "HAC(5일) 일별 차이 t" in R.describe_decision(d)


def test_retirement_uses_the_hac_statistic_too():
    # losing in long runs: the day-clustered t is far below -1, the HAC t is not
    runs = ([[-1.0, -0.6]] * 8 + [[1.0, 0.3]] * 7) * 4
    rec = S.r_stats(_trades(runs))
    assert rec["t_day"] <= -1.0 < rec["t_hac"] and rec["n"] >= 60 and rec["days"] >= 28
    d = _judge(runs, at_day=60.05)
    assert d["retire_stat"] == round(rec["t_hac"], 4) and d["verdict"] == "hold", d["rule"]
    steady = [[-1.0, 0.3], [-0.6, 0.2], [0.4, -0.3]] * 20  # losing without runs: retired on the HAC t as well
    d = _judge(steady, at_day=60.05)
    assert d["verdict"] == "retire" and "HAC t_day <= -1" in d["rule"]


def test_a_decision_stored_before_e2_still_renders():
    old = {"path": "absolute", "checkpoint": 30, "next_checkpoint": 60, "checkpoints": [14, 30, 60, 90],
           "challenger": {"n": 40, "days": 20, "avg_r": 0.1, "t_day": 2.0, "pf_r": 1.3, "max_dd_r": 4.0},
           "champion": {"n": 0}, "bar": {"min_trades": 30, "min_days": 14, "min_t_day": 3.35, "min_pf_r": 1.15}}
    text = R.describe_decision(old)
    assert "t_day +2.00" in text and "t_day≥3.35" in text and "HAC" not in text


def test_a_zero_edge_challenger_under_persistent_day_factors_is_rarely_promoted():
    """The production judge over zero-edge challengers whose day factor follows an AR(1) with phi 0.6 (the calibration
    model's worst dependence for the C2-B design), looked at through the trial as the hourly scheduler does: the
    false-promotion rate stays near its 5% target, while the C2-B rule (lag-0 statistics and thresholds) promotes
    clearly more of the same challengers."""
    cal = _cal()
    rng = np.random.default_rng(17)
    m = 500
    sc = cal.Scenario("abs", 2, 0.60, 0.3, phi=0.6)
    ch_tr, _ = cal.simulate(sc, m, rng)
    rec = cal.Record(ch_tr, m)
    old = cal.promoted({"stats": cal.checkpoint_stats(rec, None), "retire": None}, "normal",
                       cal.C2B_THRESHOLDS["normal"]["abs"], cal.C2B_THRESHOLDS["normal"]["rel"], lag=0)
    looks = sorted({c * 24 + 1 for c in R.PROMOTION_CHECKPOINTS} | set(range(5 * 24, 90 * 24, 9 * 24)))
    p, dd, r = ch_tr
    promoted = 0
    for i in range(m):
        trades = [{"r_multiple": float(r[j]), "exit_time": T0 + int(dd[j]) * MS_DAY + 3_600_000 + k, "entry_time": T0}
                  for k, j in enumerate(np.where(p == i)[0])]
        for look in looks:
            now = T0 + look * MS_HOUR
            v = judge_challenger([t for t in trades if t["exit_time"] <= now], [], ST, T0, now, [])["verdict"]
            if v != "hold":
                promoted += v == "promote"
                break
    fp = promoted / m
    assert fp <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / m)
    assert old.mean() >= fp + 0.03, (old.mean(), fp)

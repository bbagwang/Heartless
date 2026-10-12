"""C2-B: promotion that is safe under repeated looks. A challenger is judged for promotion only at fixed checkpoints after
it started (research.PROMOTION_CHECKPOINTS, FAILED_CHECKPOINTS for a failed-alpha re-enable), on its record cut at the
checkpoint, with Monte Carlo-calibrated t thresholds (tools/calibrate_promotion.py); it is retired at the end of its trial
if not promoted. Retirement stays continuous. Synthetic records only."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from heartless.config import Settings
from heartless.execution.stats import paired_daily_diff, r_stats
from heartless.learning import research as R
from heartless.learning.research import judge_challenger
from heartless.util.timeutil import MS_DAY, MS_HOUR

S = Settings(_env_file=None)
T0 = 1_780_000_000_000 // MS_DAY * MS_DAY  # a UTC midnight
ROOT = Path(__file__).resolve().parents[1]


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


def _judge(days, champ=(), at_day: float = None, new_alphas=("discovered",)):
    at = T0 + int((at_day if at_day is not None else len(days)) * MS_DAY) + MS_HOUR
    return judge_challenger(_trades(days), _trades(list(champ), "paper"), S, T0, at, list(new_alphas))


STRONG = [[1.0, 1.0, -0.5], [1.0, -0.5, 0.0], [1.0, 1.0, -0.5]]  # 3 days, t_day grows fast (mean +0.39R)
WEAK = [[1.0, -0.6], [-1.0, 0.7], [0.5, -0.5], [1.0, -1.0]]  # 4 days, mean +0.0125R


def test_the_c2b_calibration_is_kept_as_the_comparison_design():
    # E2 replaced the C2-B thresholds (HAC statistics, tests/test_e2_promotion.py); the C2-B calibration stays in
    # docs/results as history and its thresholds are what the calibration tool scores as the "before" design
    doc = json.loads((ROOT / "docs" / "results" / "c2b_promotion_calibration.json").read_text())
    ch, old = doc["chosen"], _cal().C2B_THRESHOLDS
    for cls in ("normal", "failed"):
        for path in ("abs", "rel"):
            assert {int(k): v for k, v in ch[cls][path].items()} == old[cls][path], (cls, path)
    # strictest first: most of the error budget is spent where most of the evidence is (the late checkpoints)
    for t in (old["normal"]["abs"], old["normal"]["rel"], old["failed"]["abs"], old["failed"]["rel"]):
        v = list(t.values())
        assert v == sorted(v, reverse=True) and v[-1] >= 1.8
    model = doc["model"]
    assert model["dd_scaling"] is True and model["targets"] == {"normal": 0.05, "failed": 0.02}
    for cls, target in (("normal", 0.05), ("failed", 0.02)):
        summ = doc["designs"][str(doc["chosen_delta"])][cls]["summary"]
        assert summ["false_promotion"]["max"] <= target + 0.003  # an independent run, Monte Carlo error included


def test_nothing_is_promoted_before_the_first_checkpoint():
    d = _judge(STRONG * 4, at_day=12)
    assert d["verdict"] == "hold" and d["checkpoint"] is None and d["next_checkpoint"] == 14
    assert "before the first checkpoint" in d["rule"] and "첫 판정점 전" in R.describe_decision(d)


def test_a_strong_record_is_promoted_at_the_first_checkpoint_on_its_record_up_to_it():
    days = STRONG * 5  # 15 days
    rec14 = r_stats([t for t in _trades(days) if t["exit_time"] < T0 + 14 * MS_DAY])
    assert rec14["t_day"] >= R.ABS_T_AT[14] and rec14["n"] == 42
    d = _judge(days, at_day=14.2)
    assert d["verdict"] == "promote" and d["checkpoint"] == 14 and d["next_checkpoint"] == 30
    assert d["challenger"]["n"] == 42 and d["current"]["n"] == 45 and d["bar"]["min_t_day"] == R.ABS_T_AT[14]
    assert d["rule"] == "absolute bar met at the 14-day checkpoint"


def test_between_checkpoints_the_verdict_of_the_last_one_stands():
    # weak for 14 days, then strong: the 14-day checkpoint holds, and the strong days do not count until day 30
    days = WEAK * 4 + STRONG * 6  # 16 + 18 days
    for at in (14.1, 16, 22, 29.9):
        d = _judge(days, at_day=at)
        assert d["verdict"] == "hold" and d["checkpoint"] == 14, at
        assert d["challenger"]["n"] == r_stats([t for t in _trades(days) if t["exit_time"] < T0 + 14 * MS_DAY])["n"]
        assert d["current"]["n"] > d["challenger"]["n"] or at < 14.5
    d30 = _judge(days, at_day=30.1)
    assert d30["checkpoint"] == 30 and d30["challenger"]["n"] == len(
        [t for t in _trades(days) if t["exit_time"] < T0 + 30 * MS_DAY])
    assert d30["bar"]["min_t_day"] == R.ABS_T_AT[30]


def test_the_trial_ends_at_the_last_checkpoint():
    days = WEAK * 23  # 92 days, never strong enough, never bad enough to be retired early
    d = _judge(days, at_day=89.9)
    assert d["verdict"] == "hold" and d["checkpoint"] == 60
    d = _judge(days, at_day=90.05)
    assert d["verdict"] == "retire" and d["checkpoint"] == 90 and d["next_checkpoint"] is None
    assert d["rule"] == "trial over: not promoted at the last checkpoint (90 days)"


def test_a_failed_alpha_follows_the_later_and_stricter_schedule():
    days = STRONG * 11  # 33 days
    d = _judge(days, at_day=20, new_alphas=["squeeze_breakout"])
    assert d["verdict"] == "hold" and d["checkpoint"] is None and d["next_checkpoint"] == 30
    assert d["checkpoints"] == [30, 60, 90] and d["bar"]["min_days"] == 30
    d = _judge(days, at_day=30.1, new_alphas=["squeeze_breakout"])
    assert d["checkpoint"] == 30 and d["bar"]["min_t_day"] == R.FAILED_ABS_T_AT[30] and d["failed_alphas"]
    assert d["verdict"] == "promote", d["checks"]
    assert "OOS 탈락 알파 squeeze_breakout" in R.describe_decision(d)


def test_the_relative_path_judges_the_paired_difference_at_the_checkpoint():
    champ = [[0.3, -0.3]] * 32
    better = [[1.3, 0.2], [0.6, -0.3], [1.0, 0.2]] * 11
    d = _judge(better, champ, at_day=30.1)
    assert d["path"] == "relative" and d["checkpoint"] == 30
    cut = T0 + 30 * MS_DAY
    pd_ = paired_daily_diff([t for t in _trades(better) if t["exit_time"] < cut],
                            [t for t in _trades(champ, "paper") if t["exit_time"] < cut], T0, cut - 1,
                            lag=R.PROMOTION_HAC_LAG)  # E2: the HAC paired t
    assert d["paired"]["t"] == round(pd_["t"], 4) and d["paired"]["days"] == 30
    assert d["bar"]["min_paired_t"] == R.REL_T_AT[30]
    assert d["verdict"] == ("promote" if pd_["t"] >= R.REL_T_AT[30] else "hold")


def test_retirement_stays_continuous():
    d = _judge([[-0.8, -0.6]] * 8, at_day=8)  # 16 trades, mean -0.7R: dropped before any checkpoint
    assert d["verdict"] == "retire" and d["rule"].startswith("early drop") and d["checkpoint"] is None
    losing = ([[-1.0, 0.3]] * 2 + [[1.0, -0.2]]) * 10  # 60 trades, 30 days, HAC t_day far below -1 (E2)
    d = _judge(losing, at_day=30.5)
    assert d["verdict"] == "retire" and "t_day <= -1" in d["rule"]


def test_the_absolute_drawdown_cap_grows_with_the_record():
    assert R.abs_dd_cap(10) == R.abs_dd_cap(30) == 10.0
    assert R.abs_dd_cap(120) == pytest.approx(20.0) and R.abs_dd_cap(270) == pytest.approx(30.0)
    # a long losing streak inside an otherwise strong 90-day record: over 10R deep, under the scaled cap
    days = STRONG * 30
    days[40] = [-1.0] * 12
    rec = r_stats(_trades(days))
    assert 10.0 < rec["max_dd_r"] <= R.abs_dd_cap(rec["n"])
    d = _judge(days, at_day=90.05)
    assert d["checks"][f"DD <= {R.abs_dd_cap(rec['n']):.2f}R"] is True


# --- the calibration model against the production statistics ----------------------------------------------------------
def test_the_vectorised_statistics_equal_the_production_ones():
    cal = _cal()
    rng = np.random.default_rng(3)
    for sc in (cal.Scenario("E", 2, 0.30, 0.3, 1.0), cal.Scenario("T", 3, 0.6, 0.0), cal.Scenario("abs", 1, 0.45, 0.3)):
        m = 12
        ch_tr, champ_tr = cal.simulate(sc, m, rng)
        ch, champ = cal.Record(ch_tr, m), (cal.Record(champ_tr, m) if champ_tr else None)
        pair = cal.Paired(ch, champ) if champ else None
        for i in range(m):
            def trades(tr):
                p, dd, r = tr
                return [{"r_multiple": float(r[j]), "exit_time": T0 + int(dd[j]) * MS_DAY + k + 1, "entry_time": T0}
                        for k, j in enumerate(np.where(p == i)[0])]
            a_all, b_all = trades(ch_tr), (trades(champ_tr) if champ_tr else [])
            for c in cal.CHECKPOINTS:
                cut = T0 + c * MS_DAY
                a = [t for t in a_all if t["exit_time"] < cut]
                st, d = r_stats(a), c - 1
                assert (ch.n[i, d], ch.k[i, d]) == (st["n"], st["days"])
                assert ch.t_day(d)[i] == pytest.approx(st["t_day"], abs=1e-6)
                assert ch.mean(d)[i] == pytest.approx(st["avg_r"], abs=1e-9)
                assert ch.dd(d)[i] == pytest.approx(st["max_dd_r"], abs=1e-6)
                assert (ch.pf(d)[i] == st["pf_r"]) if math.isinf(st["pf_r"]) else ch.pf(d)[i] == pytest.approx(st["pf_r"])
                if champ is not None:
                    b = [t for t in b_all if t["exit_time"] < cut]
                    assert pair.t(d)[i] == pytest.approx(paired_daily_diff(a, b, T0, cut - 1)["t"], abs=1e-6)


def test_a_zero_edge_challenger_is_rarely_promoted_by_the_production_rule():
    """The production judge over simulated zero-edge challengers (the calibration model, one realistic scenario), looked
    at repeatedly through the trial (every checkpoint and every few days in between, where the last checkpoint's
    verdict must stand): the false-promotion rate stays near its calibrated value, while the old fixed bar re-checked
    at the end of every day promoted several times as many of the same challengers."""
    cal = _cal()
    rng = np.random.default_rng(11)
    m = 400
    ch_tr, _ = cal.simulate(cal.Scenario("abs", 2, 0.45, 0.3), m, rng)
    rec = cal.Record(ch_tr, m)
    old = np.zeros(m, bool)
    for d in range(90):  # the P4 absolute bar, re-checked daily
        old |= ((rec.n[:, d] >= 30) & (rec.k[:, d] >= 14) & (rec.mean(d) > 0) & (rec.t_day(d) >= 1.5)
                & (rec.pf(d) >= 1.15) & (rec.dd(d) <= 10.0))
    looks = sorted({c * 24 + 1 for c in R.PROMOTION_CHECKPOINTS} | set(range(5 * 24, 90 * 24, 7 * 24)))
    p, dd, r = ch_tr
    promoted = 0
    for i in range(m):
        trades = [{"r_multiple": float(r[j]), "exit_time": T0 + int(dd[j]) * MS_DAY + 3_600_000 + k, "entry_time": T0}
                  for k, j in enumerate(np.where(p == i)[0])]
        verdicts = []
        for look in looks:
            now = T0 + look * MS_HOUR
            d = judge_challenger([t for t in trades if t["exit_time"] <= now], [], S, T0, now, [])
            verdicts.append(d["verdict"])
            if d["verdict"] != "hold":
                break
        promoted += verdicts[-1] == "promote"
    fp = promoted / m
    assert fp <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / m)
    assert old.mean() >= 2.5 * max(fp, 0.02), (old.mean(), fp)

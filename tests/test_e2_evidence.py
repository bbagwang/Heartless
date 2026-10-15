"""E2: the evidence registry's `stage` field and the P2 TRAIN-gate failures. Every entry says where its verdict was
reached ('oos': P1, P2 VALID, the sealed 2025 test; 'train': a P2 family whose shipped design failed the TRAIN gate);
xs_rank_mom and funding_carry_xs are recorded as failed at the TRAIN stage, so a challenger switching them on faces the
failed-alpha bar."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heartless.config import Settings
from heartless.learning import research as R
from heartless.learning import trials
from heartless.strategy import evidence as E
from heartless.strategy.params import StrategyParams

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FAILURES = {"xs_rank_mom": "xs_rank", "funding_carry_xs": "funding_carry"}


def test_every_entry_says_at_which_stage_its_verdict_was_reached():
    reg = E.registry()
    assert reg["schema_version"] >= 2 and set(reg["stages"]) == set(E.STAGES)
    for name, e in reg["alphas"].items():
        assert e["stage"] in E.STAGES, name
        if e["stage"] == "oos":
            assert e.get("oos_window"), name  # P1, P2 VALID or the sealed 2025 test
        else:  # a TRAIN-stage verdict never claims an out-of-sample window
            assert "oos_window" not in e and e["judgement"] == "p2" and e["rule"] == "p2-train-gate", name
    assert {a for a, e in reg["alphas"].items() if e["stage"] == "train"} == set(TRAIN_FAILURES)


def test_the_train_gate_failures_are_recorded_with_their_train_numbers():
    p2 = json.loads((ROOT / "docs" / "results" / "p2_decisions.json").read_text())
    gate = p2["train_gate"]
    for name, fam in TRAIN_FAILURES.items():
        e = E.registry()["alphas"][name]
        assert e["verdict"] == "failed" and e["family"] == fam and p2["families"][fam]["final_candidates"] == []
        tr = e["train"]
        assert tr["n_family"] == p2["families"][fam]["n_family"]
        assert tr["t_day_bar"] == pytest.approx(trials.threshold(tr["n_family"]))
        res = json.loads((ROOT / tr["file"]).read_text())
        stress = json.loads((ROOT / tr["stress_file"]).read_text())
        s = res["summary"]
        assert (res["alpha"], res["overrides"], res["trial"]["period"]) == (name, [], "TRAIN")  # the shipped defaults
        assert (e["n"], e["avg_r"], e["t_day"]) == (tr["n"], tr["avg_r"], tr["t_day"]) == (s["n"], s["avg_r"], s["t_day"])
        assert e["stress_avg_r"] == tr["stress_avg_r"] == stress["summary"]["avg_r"]
        assert stress["trial"]["config_hash"] == res["trial"]["config_hash"] == tr["config_hash"]
        assert e["years_positive"] == sum(1 for v in res["by_year"].values() if v["avg_r"] > 0) and e["years"] == 3
        # the entry describes the shipped design: today's default configuration is the one the TRAIN run evaluated
        cfg = trials.lab_config(StrategyParams.default().to_dict(), name, res["trial"]["config"]["symbols"])
        assert trials.config_hash(cfg) == tr["config_hash"]
        assert e["param_schema"] == E.param_fingerprint(name)
        checks = {"t_day >= 3.0": tr["t_day"] >= tr["t_day_bar"], "avg_r >= 0.05": tr["avg_r"] >= gate["avg_r_ge"],
                  "years_positive >= 2": tr["years_positive"] >= gate["years_positive_ge"],
                  "stress_avg_r > 0": tr["stress_avg_r"] > 0}
        assert e["checks"] == checks and not all(checks.values())
        assert e["source"] == "docs/RESEARCH_P2.md" and (ROOT / e["source"]).exists()
    best = E.registry()["alphas"]["xs_rank_mom"]["best_lab"]  # the family's best lab run (another configuration)
    fam = p2["families"]["xs_rank"]["best_lab"]
    assert (best["file"], best["n"], best["avg_r"], best["t_day"]) == (fam["file"], fam["n"], fam["avg_r"], fam["t_day"])
    fc = p2["families"]["funding_carry"]["best_lab"]  # funding_carry's best lab run is the shipped configuration
    assert E.registry()["alphas"]["funding_carry_xs"]["train"]["file"] == fc["file"]


def test_a_train_gate_failure_counts_as_failed_everywhere():
    for name in TRAIN_FAILURES:
        assert E.verdict(name) == "failed" and E.stage(name) == "train" and E.expected_r(name) == 0.0
        assert E.short(name).startswith("TRAIN 관문 탈락 (2022-01-13..2024-12-31")
    assert E.stage("htf_trend") == "oos" and E.stage("discovered") is None
    assert E.failed(["discovered", "xs_rank_mom", "funding_carry_xs"]) == ["xs_rank_mom", "funding_carry_xs"]
    # a challenger switching one of them on: the failed-alpha schedule and bar
    s = Settings(_env_file=None)
    assert R.checkpoints(E.failed(["xs_rank_mom"])) == R.FAILED_CHECKPOINTS
    bar = R.promotion_bar(s, "absolute", E.failed(["funding_carry_xs"]), 30)
    assert bar["min_days"] == R.FAILED_MIN_DAYS and bar["min_t_hac"] == R.FAILED_ABS_T_AT[30]
    assert bar["failed_alphas"] == ["funding_carry_xs"]


def test_an_entry_without_a_stage_is_an_oos_verdict_and_a_bad_stage_is_refused(monkeypatch):
    data = json.loads(json.dumps(E.registry()))
    del data["alphas"]["htf_trend"]["stage"]  # an entry written before the field existed
    monkeypatch.setattr(E, "registry", lambda: data)
    assert E.stage("htf_trend") == "oos" and E.short("htf_trend").startswith("OOS 탈락")
    monkeypatch.undo()
    bad = json.loads(json.dumps(E.registry()))
    bad["alphas"]["htf_trend"]["stage"] = "valid"

    class _Res:
        def __init__(self, text):
            self.text = text

        def joinpath(self, _):
            return self

        def open(self, *a, **k):
            import io

            return io.StringIO(self.text)

    E.registry.cache_clear()
    monkeypatch.setattr(E.resources, "files", lambda _pkg: _Res(json.dumps(bad)))
    try:
        with pytest.raises(ValueError, match="stage"):
            E.registry()
    finally:
        monkeypatch.undo()
        E.registry.cache_clear()
    assert E.stage("htf_trend") == "oos"

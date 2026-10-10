"""P4 evidence registry (heartless/strategy/evidence.json + evidence.py)."""
import json
from pathlib import Path

import pytest

from heartless.strategy import evidence as E
from heartless.strategy.alphas import ALPHA_BY_NAME

ROOT = Path(__file__).resolve().parents[1]


def test_registry_agrees_with_the_p1_decision_table():
    p1 = json.loads((ROOT / "docs" / "results" / "p1_decisions.json").read_text())
    reg = E.registry()["alphas"]
    assert set(reg) == set(p1["alphas"])  # the nine judged alphas, nothing else
    for name, d in p1["alphas"].items():
        e, inp = reg[name], d["inputs"]
        assert e["verdict"] == ("passed" if d["after"] else "failed"), name
        assert all(d["checks"].values()) == (e["verdict"] == "passed")
        for k in ("n", "avg_r", "t_day", "years_positive", "stress_avg_r"):
            assert e[k] == inp[k], (name, k)
        assert e["years"] == len(inp["years"]) and e["oos_window"] == "2022-01-13..2024-12-31"
        assert e["source"] == "docs/OOS_2022_2024.md" and (ROOT / e["source"]).exists()
    assert all(e["verdict"] == "failed" for e in reg.values())  # P1: no alpha passed


def test_registry_entries_describe_the_current_designs():
    # a redesign must come with a re-judged entry: until then the alpha would silently count as 'untested'
    for name, e in E.registry()["alphas"].items():
        assert name in ALPHA_BY_NAME, name
        assert e["param_schema"] == E.param_fingerprint(name), f"{name}: parameter schema changed since it was judged"


def test_verdicts_defaults_and_expectation():
    assert E.verdict("htf_trend") == "failed" and E.verdict("discovered") == "untested"
    assert E.verdict("no_such_alpha") == "untested" and E.entry("no_such_alpha") is None
    assert E.failed(["discovered", "momentum_burst", "squeeze_breakout"]) == ["momentum_burst", "squeeze_breakout"]
    assert E.expected_r("htf_trend") == 0.0  # a failed alpha is expected to earn nothing, not its negative OOS mean
    assert E.summary(["htf_trend", "discovered"]) == {"failed": 1, "passed": 0, "untested": 1}
    assert "OOS 탈락" in E.short("htf_trend") and E.short("discovered") == "OOS 미검증"


def test_stale_and_passed_entries(monkeypatch):
    data = json.loads(json.dumps(E.registry()))
    data["alphas"]["htf_trend"]["param_schema"] = "000000000000"  # judged with another parameter schema
    data["alphas"]["squeeze_breakout"].update(verdict="passed", avg_r=0.12)
    monkeypatch.setattr(E, "registry", lambda: data)
    assert E.verdict("htf_trend") == "untested" and E.entry("htf_trend") is None
    assert E.verdict("squeeze_breakout") == "passed" and E.expected_r("squeeze_breakout") == pytest.approx(0.12)


def test_registry_is_package_data():
    text = (ROOT / "pyproject.toml").read_text()
    assert '"heartless.strategy" = ["evidence.json"]' in text
    assert (ROOT / "heartless" / "strategy" / "evidence.json").exists()

"""P2a trial ledger: append-only lines, N per family over distinct configurations, and the max(3, sqrt(2 ln N)) bar."""
import json
import math
from datetime import datetime, timezone

import pytest

from heartless.core.store import Store
from heartless.learning import discovery as D
from heartless.learning import seal, trials


def _ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    """Like the conftest seal-ledger fixture: tests never append to the real docs/results/p2_trials.jsonl."""
    path = tmp_path / "p2_trials.jsonl"
    monkeypatch.setattr(trials, "LEDGER", path)
    return path


def test_threshold_is_max_3_and_sqrt_2_ln_n():
    assert trials.threshold(0) == trials.threshold(1) == trials.threshold(90) == 3.0
    assert trials.threshold(10_000) == pytest.approx(math.sqrt(2 * math.log(10_000)))
    assert trials.threshold(50_000) == D.multiple_testing_t(50_000)


def test_record_load_and_counts_per_family(ledger):
    train = (_ms(2022, 1, 13), _ms(2025, 1, 1) - 1)
    valid = (_ms(2026, 1, 1), _ms(2026, 10, 1) - 1)
    a = {"tool": "lab", "alpha": "xs_mom", "params": {"k": 1}}
    b = {"tool": "lab", "alpha": "xs_mom", "params": {"k": 2}}
    e1 = trials.record("lab", "xs_mom", a, train, n=120, avg_r=0.05123456, t_day=1.234567)
    trials.record("lab", "xs_mom", a, valid, n=40, avg_r=-0.1, t_day=-0.5)  # same config on VALID: not a new trial
    trials.record("lab", "xs_mom", b, train, n=99, avg_r=float("nan"), t_day=None)
    trials.record("discover", "rules_1h", {"tool": "discover", "grids": ["1h"]}, train, count=12_000)
    trials.record("discover", "rules_1h", {"tool": "discover", "grids": ["1h"]}, train, count=12_000)  # a rerun
    trials.record("discover", "rules_1h", {"tool": "discover", "grids": ["1h", "1d"]}, train, count=8_000)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 6 and all(json.loads(x)["commit"] == seal.git_commit() for x in lines)
    assert e1["period"] == "TRAIN" and e1["window"] == ["2022-01-13 00:00", "2024-12-31 23:59"]
    assert e1["avg_r"] == 0.05123 and e1["t_day"] == 1.235 and json.loads(lines[2])["avg_r"] is None
    assert json.loads(lines[1])["period"] == "VALID"
    fams = trials.summary(trials.load())
    assert fams["xs_mom"]["lines"] == 3 and fams["xs_mom"]["configs"] == 2 and fams["xs_mom"]["n_tests"] == 2
    assert fams["xs_mom"]["threshold"] == 3.0
    assert fams["rules_1h"]["n_tests"] == 20_000 and fams["rules_1h"]["threshold"] == round(trials.threshold(20_000), 3)
    assert trials.family_tests("rules_1h") == 20_000 and trials.family_tests("nope") == 0
    text = trials.report(trials.load())
    assert "xs_mom" in text and "rules_1h" in text and "20000" in text
    one = trials.report(trials.load(), "xs_mom")
    assert "rules_1h" not in one and one.count(e1["config_hash"]) == 2
    with pytest.raises(ValueError):
        trials.record("lab", " ", a, train)


def test_lab_config_identity_ignores_window_and_costs_but_not_code_or_params():
    from heartless.strategy.params import StrategyParams

    p = StrategyParams.default().to_dict()
    c1 = trials.lab_config(p, "htf_trend", ["BTCUSDT", "ETHUSDT"])
    c2 = trials.lab_config(p, "htf_trend", ["ETHUSDT", "BTCUSDT"])
    assert trials.config_hash(c1) == trials.config_hash(c2) and c1["code"] == trials.code_hash(["htf_trend"])
    p2 = json.loads(json.dumps(p))
    p2["alphas"]["htf_trend"]["sl_atr"] = 2.0
    p2["alphas"]["trend_pullback"]["sl_atr"] = 9.0  # not evaluated in a --alpha htf_trend run: irrelevant
    assert trials.config_hash(trials.lab_config(p2, "htf_trend", ["BTCUSDT", "ETHUSDT"])) != trials.config_hash(c1)
    p3 = json.loads(json.dumps(p))
    p3["alphas"]["trend_pullback"]["sl_atr"] = 9.0
    assert trials.config_hash(trials.lab_config(p3, "htf_trend", ["BTCUSDT", "ETHUSDT"])) == trials.config_hash(c1)
    assert trials.code_hash(["htf_trend"]) != trials.code_hash(["momentum_burst"])


def test_discover_cli_tag_records_the_rules_evaluated_and_uses_the_family_n(tmp_path, monkeypatch, capsys, ledger):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    Store(tmp_path / "heartless.db").close()
    trials.record("discover", "fam", {"earlier": 1}, (_ms(2022, 1, 1), _ms(2022, 6, 1)), count=5_000)
    calls = []

    def fake_mine(db, syms, train, valid, tf, cfg, **k):
        calls.append((tf, k))
        best = {"n": 300, "avg_r": 0.1, "t_ex": 2.5}
        return {"tested": 7_000, "n_tests": 7_000 + k.get("n_prior", 0), "t_bar": 4.0, "passed": [], "results": [],
                "seconds": 1.0, "rows": {"train": 10}, "tf": tf, "best_train": best}

    monkeypatch.setattr(D, "mine", fake_mine)
    cli.main(["discover", "--symbols", "BTCUSDT", "--grid", "1h,1d", "--templates", "all", "--tag", "fam",
              "--train", "2022-01-15:2025-01-01", "--valid", "2026-01-01:2026-10-01"])
    tf, k = calls[-1]
    assert tf == "1h,1d" and k["templates"] == "all" and k["n_prior"] == 5_000 and k["context"] is None
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]["tool"] == "discover" and rows[-1]["count"] == 7_000
    assert rows[-1]["n"] == 300 and rows[-1]["t_day"] == 2.5 and rows[-1]["period"] == "TRAIN"
    assert trials.family_tests("fam") == 12_000
    err = capsys.readouterr().err
    assert "family N=12000" in err
    cli.main(["trials"])
    out = capsys.readouterr().out
    assert "fam" in out and "12000" in out
    cli.main(["trials", "--family", "fam"])
    assert rows[-1]["config_hash"] in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["discover", "--symbols", "BTCUSDT", "--grid", "4h", "--train", "2022-01-15:2025-01-01",
                  "--valid", "2026-01-01:2026-10-01"])

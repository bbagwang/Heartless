"""F-pre-A trial ledger: Wave F period labels (TRAIN-F / TRAIN-A / TRAIN-B / VALID-F / FWD-k) derived from the window,
the execution mode `exec` inside the configuration hash with taker as the unhashed default (every P2 hash reproduced),
and stress / VALID / FORWARD reruns of a configuration that never raise N."""
import json
from pathlib import Path

import pytest

from heartless.learning import trials
from heartless.strategy.params import StrategyParams

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "results" / "p2_trials.jsonl"
REGISTRATION = ROOT / "docs" / "results" / "wave_f_registration.json"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "p2_trials.jsonl"
    monkeypatch.setattr(trials, "LEDGER", path)
    return path


def test_every_p2_hash_is_reproduced_with_the_exec_default():
    """The committed ledger: every stored configuration hashes to its stored hash. The P2-era lines (135, recorded
    before the field existed, so without a wave_f label) carry no exec; a Wave F line may carry one (the neutral frame
    writes exec="taker"). A taker configuration keeps its hash with an explicit exec=taker, and a maker variant of it
    is a different configuration."""
    lines = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    p2 = [e for e in lines if "wave_f" not in e]
    assert len(p2) >= 135
    for e in lines:
        cfg = e["config"]
        if "wave_f" not in e:
            assert "exec" not in cfg
        assert trials.config_hash(cfg) == e["config_hash"]
        mode = trials.exec_of(cfg)
        if any(m != "taker" for m in (mode.values() if isinstance(mode, dict) else [mode])):
            assert trials.config_hash({**cfg, "exec": "taker"}) != e["config_hash"]
            continue
        assert trials.config_hash({**cfg, "exec": "taker"}) == e["config_hash"]
        assert trials.config_hash({**cfg, "exec": {"x": "taker"}}) == e["config_hash"]
        assert trials.config_hash({**cfg, "exec": "maker_m1"}) != e["config_hash"]
    # the P2 families' N are those of ROADMAP 7.7.2 (n_for_gate), unchanged by the new code
    s = trials.summary(p2)
    assert {f: s[f]["n_tests"] for f in ("ts_trend", "positioning_flow", "vol_breakout", "discovery_daily",
                                         "xs_rank", "funding_carry")} == {
        "ts_trend": 27, "positioning_flow": 21, "vol_breakout": 23, "discovery_daily": 105068, "xs_rank": 20,
        "funding_carry": 7}


def test_lab_config_hashes_exec_only_when_not_taker():
    p = StrategyParams.default().to_dict()
    base = trials.lab_config(p, "htf_trend", ["BTCUSDT"])
    assert "exec" not in base
    assert trials.lab_config(p, "htf_trend", ["BTCUSDT"], exec="taker") == base
    maker = trials.lab_config(p, "htf_trend", ["BTCUSDT"], exec="maker_m1")
    assert maker["exec"] == {"htf_trend": "maker_m1"} and trials.config_hash(maker) != trials.config_hash(base)
    # the engine setting params["exec"][alpha] (ROADMAP 7.7.2) is read when the params carry it
    with_exec = trials.lab_config({**p, "exec": {"htf_trend": "maker_m2", "other": "maker_m1"}}, "htf_trend",
                                  ["BTCUSDT"])
    assert with_exec["exec"] == {"htf_trend": "maker_m2"}
    with pytest.raises(ValueError, match="unknown execution mode"):
        trials.lab_config(p, "htf_trend", ["BTCUSDT"], exec="post_only")
    # a topN universe is identified by its rule, not by the symbols a window's 1m data happened to have
    top = trials.lab_config(p, "htf_trend", ["BTCUSDT", "ETHUSDT"], universe="top40")
    assert top["symbols"] == "top40" and trials.lab_config(p, "htf_trend", ["SOLUSDT"], universe="top40") == top


def test_wave_f_periods_follow_the_registration():
    from heartless.learning.lab import parse_when

    reg = json.loads(REGISTRATION.read_text())
    for name in ("TRAIN-F", "TRAIN-A", "TRAIN-B", "VALID-F"):
        w = reg["windows"][name]
        a = parse_when(w["start"])
        b = parse_when(w["end_inclusive"]) + 86_400_000  # exclusive
        assert trials.WAVE_F[name] == (a, b), name
    for blk in reg["forward_seal"]["blocks"]:
        k = int(blk["id"][4:])
        assert trials.fwd_span(k) == (blk["start_ms"], blk["end_ms_exclusive"] - 1)
        assert trials.wave_f_period(blk["start_ms"], blk["end_ms_exclusive"] - 1) == blk["id"]
    assert trials.FWD_ORIGIN == reg["forward_seal"]["blocks"][0]["start_ms"]
    W = trials.WAVE_F
    assert trials.wave_f_period(W["TRAIN-F"][0], W["TRAIN-F"][1] - 1) == "TRAIN-F"
    assert trials.wave_f_period(W["TRAIN-A"][0], W["TRAIN-A"][1] - 1) == "TRAIN-A"
    assert trials.wave_f_periods(W["TRAIN-B"][0], W["TRAIN-B"][1] - 1) == ["TRAIN-B", "TRAIN-F"]
    assert trials.wave_f_period(W["VALID-F"][0], W["VALID-F"][1] - 1) == "VALID-F"
    assert trials.wave_f_period(trials.fwd_span(2)[0], trials.fwd_span(4)[1]) == "FWD-02..FWD-04"
    # outside every partition: the 2022 warm-up days, the 2026 gap before VALID-F, a window across TRAIN and VALID
    assert trials.wave_f_period(W["TRAIN-F"][0] - 86_400_000, W["TRAIN-A"][1] - 1) == "other"
    assert trials.wave_f_period(W["TRAIN-B"][1], W["VALID-F"][1] - 1) == "other"
    assert trials.wave_f_period(W["TRAIN-B"][0], W["VALID-F"][1] - 1) == "other"
    assert trials.fwd_block(trials.FWD_ORIGIN - 1) == 0 and trials.fwd_block(trials.FWD_ORIGIN) == 1


def test_reruns_keep_n_and_lines_carry_the_labels(ledger):
    W = trials.WAVE_F
    cfg = {"screen": "carry", "k": 3, "A_in": 0.2, "universe": "top40"}
    train = (W["TRAIN-F"][0], W["TRAIN-F"][1] - 1)
    e = trials.record("screen", "carry_neutral", cfg, train, n=10, t_day=1.0)
    assert e["wave_f"] == "TRAIN-F" and e["period"] == "other" and e["exec"] == "taker"
    trials.record("screen", "carry_neutral", cfg, train, n=10, t_day=0.5, extra={"stress": True})
    trials.record("screen", "carry_neutral", cfg, (W["VALID-F"][0], W["VALID-F"][1] - 1), n=4)
    trials.record("screen", "carry_neutral", {**cfg, "exec": "taker"}, trials.fwd_span(2), n=2)
    trials.record("hourly", "carry_neutral", cfg, (W["TRAIN-B"][0], W["TRAIN-B"][1] - 1), n=5)
    assert trials.family_tests("carry_neutral") == 1  # one configuration, five evaluations
    trials.record("neutral", "carry_neutral", {**cfg, "exec": "maker"}, train, n=10)
    assert trials.family_tests("carry_neutral") == 2  # the maker variant is a new trial
    fam = trials.summary(trials.load())["carry_neutral"]
    assert fam["wave_f"] == {"TRAIN-F": 3, "VALID-F": 1, "FWD-02": 1, "TRAIN-B": 1}
    assert fam["tools"] == {"screen": 4, "hourly": 1, "neutral": 1} and fam["configs"] == 2
    rep = trials.report(trials.load(), "carry_neutral")
    assert "FWD-02" in rep and "exec=maker" in rep and "stress" in rep
    with pytest.raises(ValueError, match="unknown execution mode"):
        trials.record("screen", "carry_neutral", {**cfg, "exec": "limit"}, train)
    with pytest.raises(ValueError, match="unknown trial tool"):
        trials.record("gate", "carry_neutral", cfg, train)

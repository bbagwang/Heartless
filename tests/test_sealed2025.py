"""Sealed-2025 confirmation test (docs/SEALED_2025.md, pre-registered in docs/ROADMAP.md section 6 on 2026-10-11): the
decision file applies the pre-registered rule to its inputs, the lab result files, the trial ledger and the seal ledger
back it, the evidence registry and the shipped default follow it, and the spent seal no longer refuses 2025."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from heartless.config import Settings
from heartless.core.store import Store
from heartless.execution.stats import day_clustered_t
from heartless.learning import lab, seal, trials
from heartless.strategy import evidence as E
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.params import StrategyParams

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
S25, E25 = 1_735_689_600_000, 1_767_225_600_000  # 2025-01-01 .. 2026-01-01 UTC
CONFIG = "a8356702eb374c77"  # vol_breakout_dbox with comp_q=1.0: the TRAIN and VALID configuration


def _dec() -> dict:
    return json.loads((RESULTS / "p2_sealed2025_decision.json").read_text())


def _trades_from_daily(rows) -> list[dict]:
    """Trades with the same per-day counts and R sums as a [day, n, sum R] table: day_clustered_t only sees those."""
    from datetime import datetime, timezone

    out = []
    for day, n, s in rows:
        t = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000) + 3_600_000
        out += [{"exit_time": t, "r_multiple": s / n} for _ in range(n)]
    return out


def test_decision_applies_the_preregistered_rule():
    d = _dec()
    s, ss, v = d["inputs"]["sealed2025"], d["inputs"]["sealed2025_stress"], d["inputs"]["valid"]
    assert d["rule"]["pooled_bar"] == 1.65 and d["preregistration"]["commit"] == "ae0b511"
    daily = d["pooled"]["daily"]
    assert sum(n for _, n, _ in daily["valid"]) == v["n"] and sum(n for _, n, _ in daily["sealed2025"]) == s["n"]
    assert not {r[0] for r in daily["valid"]} & {r[0] for r in daily["sealed2025"]}  # disjoint exit days
    pooled = _trades_from_daily(daily["valid"] + daily["sealed2025"])
    assert len(pooled) == d["pooled"]["n"] == v["n"] + s["n"]
    assert day_clustered_t(pooled) == pytest.approx(d["pooled"]["t_day_exact"], abs=1e-6)
    assert day_clustered_t(_trades_from_daily(daily["valid"])) == pytest.approx(v["check"]["t_day_exact"], abs=1e-6)
    checks = {"sealed_avg_r_gt_0": s["avg_r"] > 0, "sealed_stress_avg_r_gt_0": ss["avg_r"] > 0,
              "pooled_t_day_ge_1_65": d["pooled"]["t_day_exact"] >= 1.65}
    assert d["checks"] == checks
    assert d["verdict"] == ("PASS" if all(checks.values()) else "FAIL")


def test_result_files_and_ledgers_back_the_decision():
    d = _dec()
    ledger = {(e["config_hash"], e["utc"]): e for e in trials.load()}
    for key, stress in (("valid", False), ("sealed2025", False), ("sealed2025_stress", True)):
        inp = d["inputs"][key]
        res = json.loads((ROOT / inp["file"]).read_text())
        assert (res["summary"]["n"], res["summary"]["avg_r"], res["summary"]["t_day"]) == \
            (inp["n"], inp["avg_r"], inp["t_day"]), key
        assert res["trial"]["config_hash"] == CONFIG and res["alpha"] == "vol_breakout_dbox", key
        assert res["overrides"] == ["vol_breakout_dbox.comp_q=1.0"] and res["git_dirty"] is False, key
        assert (res["trial"]["config_hash"], res["trial"]["utc"]) in ledger, key
        assert res["trial"]["stress"] is stress, key
        if key != "valid":
            assert res["window"]["start_ms"] == S25 and res["window"]["end_ms"] == E25 - 1, key
            assert res["seal"]["unsealed"] == ["2025"] and res["summary"]["boundary"] == 0, key
    # the 2025 runs re-ran a configuration the family had already counted: N did not grow. The two exam runs are
    # identified by the ledger lines their result files carry, not by their window: 2025 is ordinary history now, so
    # later vol_breakout research may append more 2025 lines (and later trials of any kind) to the ledger
    fam = [e for e in trials.load() if e["family"] == "vol_breakout"]
    utcs = {json.loads((ROOT / d["inputs"][k]["file"]).read_text())["trial"]["utc"]
            for k in ("sealed2025", "sealed2025_stress")}
    runs = [e for e in fam if e["config_hash"] == CONFIG and e["utc"] in utcs]
    upto = [e for e in fam if e["utc"] <= max(e["utc"] for e in runs)]
    n_after = trials.summary(upto, "vol_breakout")["vol_breakout"]["n_tests"]
    n_before = trials.summary([e for e in upto if e not in runs], "vol_breakout")["vol_breakout"]["n_tests"]
    assert len(runs) == 2 and all(e["window"] == ["2025-01-01 00:00", "2025-12-31 23:59"] for e in runs)
    assert n_after == n_before == d["configuration"]["family_n_after"] == 23
    # every opening of the sealed period is in the seal ledger, then the line that spends the seal (a later sealed
    # period's lines may follow it; a spent 2025 records nothing more)
    rows = [json.loads(x) for x in (RESULTS / "seal_ledger.jsonl").read_text().splitlines()]
    opens = [i for i, r in enumerate(rows) if r["period"] == "2025" and not r.get("event")]
    spent = [i for i, r in enumerate(rows) if r["period"] == "2025" and r.get("event") == "spent"]
    assert [rows[i]["argv"][1] for i in opens] == ["lab", "lab", "backtest"]
    assert all("--unseal" in rows[i]["argv"] for i in opens)
    assert len(spent) == 1 and spent[0] > max(opens)
    joint = json.loads((RESULTS / "p2_sealed2025_joint.json").read_text())
    assert joint["enabled"] == ["vol_breakout_dbox"] and joint["seal"]["unsealed"] == ["2025"]
    assert joint["window"]["start_ms"] == S25 and joint["window"]["end_ms"] == E25 - 1 and len(joint["symbols"]) == 20


def test_registry_and_shipped_default_follow_the_decision():
    d = _dec()
    e = E.registry()["alphas"]["vol_breakout_dbox"]
    s, ss = d["inputs"]["sealed2025"], d["inputs"]["sealed2025_stress"]
    blk = e["sealed2025"]
    assert (blk["n"], blk["avg_r"], blk["t_day"], blk["stress_avg_r"]) == (s["n"], s["avg_r"], s["t_day"], ss["avg_r"])
    assert blk["pooled_valid_2025"]["t_day"] == d["pooled"]["t_day"] and blk["result"] == d["verdict"].lower()
    passed = d["verdict"] == "PASS"
    assert E.verdict("vol_breakout_dbox") == ("passed" if passed else "failed")
    assert E.expected_r("vol_breakout_dbox") == (d["expected_r_if_passed"] if passed else 0.0)
    assert ALPHA_BY_NAME["vol_breakout_dbox"].enabled_by_default is passed
    assert StrategyParams.default().enabled["vol_breakout_dbox"] is passed
    assert "sealed2025" in E.registry()["judgements"]


def test_the_2025_seal_is_spent():
    """2025 is ordinary history now: no path refuses it, `--unseal 2025` (the exam commands) is accepted but opens and
    records nothing, and a typo is still an error. The mechanism itself is tested with synthetic periods
    (tests/test_p0_seal.py)."""
    assert "2025" not in seal.SEALED and seal.SPENT["2025"] == (S25, E25)
    seal.check_window(S25, E25 - 1)
    assert seal.clip_window(S25, E25 - 1) == (S25, E25 - 1)
    assert seal.readable_until(S25 - 1, S25 + 3_600_000) == S25 + 3_600_000
    seal.check_window(S25, E25 - 1, unseal=["2025"])
    assert seal.opened(S25, E25 - 1, ["2025"]) == [] and seal.status(S25, E25 - 1, ["2025"])["clean"]
    with pytest.raises(ValueError):
        seal.check_window(S25, E25 - 1, unseal=["2O25"])


def test_lab_cli_on_2025_neither_refuses_nor_records(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(settings.db_path).close()
    ran = []
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: ran.append((start, end, k)) or
                        lab.LabResult("v", None, start, end, {**lab.summarize([])}, {}, {}, {}, chunks=[(start, end)]))
    base = dict(symbols="BTCUSDT", years="2025", start=None, end=None, split="train", params=None, set=[],
                stress=False, meta=False, chunk_days=None, alpha=None, workers=1, json=False, trades_out=None,
                out=None, by_side=False, argv=["heartless", "lab"])
    lab.main_cli(settings, SimpleNamespace(**base, unseal=[]))
    lab.main_cli(settings, SimpleNamespace(**base, unseal=["2025"]))
    assert [r[:2] for r in ran] == [(S25, E25 - 1)] * 2 and all(r[2]["unseal"] == [] for r in ran)
    assert "UNSEALING" not in capsys.readouterr().err and not Path(seal.LEDGER).exists()

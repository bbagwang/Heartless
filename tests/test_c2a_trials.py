"""C2-A trial ledger checks: --stress and VALID reruns of a tagged lab configuration keep N, screening scripts record
with tool="screen" and share the family's N with lab and discover, and the documented screening template runs."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.learning import lab, trials
from heartless.strategy.params import StrategyParams

DAY = 86_400_000
INLINE = SimpleNamespace(map=map)
DOC = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_TOOLS.md"


def _ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


TRAIN = (_ms(2022, 1, 13), _ms(2025, 1, 1) - 1)
VALID = (_ms(2026, 1, 13), _ms(2026, 10, 1) - 1)


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "p2_trials.jsonl"
    monkeypatch.setattr(trials, "LEDGER", path)
    return path


@pytest.fixture
def data_dir(tmp_path):
    """One symbol with a TRAIN-period span (December 2024) and a VALID-period span (January 2026)."""
    d = tmp_path / "data"
    d.mkdir()
    st = Store(d / "heartless.db")
    rng = np.random.default_rng(3)
    for a in (_ms(2024, 12, 1), _ms(2026, 1, 1)):
        n = 30 * 1440
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
        prev = np.concatenate(([close[0]], close[:-1]))
        st.save_candles("BTCUSDT", [Candle(a + i * 60_000, float(prev[i]), float(max(prev[i], close[i]) * 1.0003),
                                           float(min(prev[i], close[i]) * 0.9997), float(close[i]), 10.0,
                                           10.0 * float(close[i]), 3, 5.0, a + i * 60_000 + 59_999) for i in range(n)])
    st.close()
    return d


def _args(**kw):
    base = dict(symbols="BTCUSDT", universe=None, tag="xs_rank", years=None, start="2024-12-20", end="2024-12-28",
                split="train", unseal=[], params=None, set=[], stress=False, meta=False, chunk_days=None,
                alpha="trend_pullback", workers=1, json=False, trades_out=None, out=None, by_side=False,
                argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_stress_and_valid_reruns_of_a_tagged_configuration_keep_n(data_dir, ledger, monkeypatch, capsys):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(data_dir), CONTEXT_AUTO_BUILD=False)
    real = lab.evaluate
    monkeypatch.setattr(lab, "evaluate", lambda *a, **k: real(*a, **{**k, "pool": INLINE}))
    lab.main_cli(settings, _args())
    lab.main_cli(settings, _args(stress=True))
    lab.main_cli(settings, _args(start="2026-01-15", end="2026-01-25"))
    lab.main_cli(settings, _args(start="2026-01-15", end="2026-01-25", stress=True))
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [r["period"] for r in rows] == ["TRAIN", "TRAIN", "VALID", "VALID"]
    assert [r["stress"] for r in rows] == [False, True, False, True]
    assert len({r["config_hash"] for r in rows}) == 1 and trials.family_tests("xs_rank") == 1
    assert trials.summary(trials.load())["xs_rank"]["threshold"] == 3.0
    lab.main_cli(settings, _args(meta=True))  # another configuration (meta-labeling on) is a new trial
    assert trials.family_tests("xs_rank") == 2
    assert "family N=2" in capsys.readouterr().err


def test_screening_scripts_share_the_family_n_with_lab_and_discover(ledger):
    lab_cfg = trials.lab_config(StrategyParams.default().to_dict(), "trend_pullback", ["BTCUSDT"])
    trials.record("lab", "xs_rank", lab_cfg, TRAIN, n=300, avg_r=0.01, t_day=0.4)
    for k in (24, 72, 168):
        e = trials.record("screen", "xs_rank", {"screen": "persist", "lookback_h": k}, TRAIN, 120, 0.02, 1.1)
        assert e["tool"] == "screen" and e["n"] == 120 and e["period"] == "TRAIN"  # n positional (5th argument)
    trials.record("screen", "xs_rank", {"screen": "persist", "lookback_h": 24}, VALID, n=40)  # the same config
    trials.record("screen", "xs_rank", {"screen": "grid", "rules": 64}, TRAIN, count=64)  # a batch of rules
    trials.record("discover", "xs_rank", {"tool": "discover", "grids": ["1d"]}, TRAIN, count=2_000)
    fam = trials.summary(trials.load())["xs_rank"]
    assert fam["n_tests"] == 1 + 3 + 64 + 2_000 == trials.family_tests("xs_rank")
    assert fam["tools"] == {"lab": 1, "screen": 5, "discover": 1} and fam["periods"] == {"TRAIN": 6, "VALID": 1}
    assert fam["threshold"] == round(trials.threshold(2_068), 3) > 3.0
    assert "screen" in trials.report(trials.load(), "xs_rank")
    with pytest.raises(ValueError, match="unknown trial tool"):
        trials.record("notebook", "xs_rank", {}, TRAIN)


def test_the_documented_screening_template_runs_and_records_one_line_per_configuration(ledger):
    text = DOC.read_text(encoding="utf-8")
    block = re.search(r"\*\*스크리닝 스크립트\.\*\*.*?```python\n(.*?)```", text, re.S)
    assert block, "screening example missing from docs/RESEARCH_TOOLS.md"
    calls = []

    def my_screen(cfg, a, b):
        calls.append((cfg["lookback_h"], a, b))
        return [{"r_multiple": 0.1 * (i % 3 - 1), "exit_time": a + i * DAY, "pnl": 1.0} for i in range(30)]

    ns = {"my_screen": my_screen, "print": lambda *a: None}
    exec(compile(block.group(1), str(DOC), "exec"), ns)
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(rows) == len(calls) == 3 and {r["tool"] for r in rows} == {"screen"}
    assert all(r["family"] == "xs_rank" and r["period"] == "TRAIN" and r["n"] == 30 for r in rows)
    assert calls[0][1:] == TRAIN and ns["n_family"] == 3

"""E2: the research cycle's warm-up, its alpha selection and its cost bounds.

* The walk-forward worker loads the longest lookback of the alphas it tunes (+ the lab's margin) BEFORE its window:
  the window used to be its own warm-up, so a long-lookback alpha saw a fraction of its history (or none) in research.
* The stored history (DATA_RETENTION_DAYS) is raised to cover the window plus that warm-up.
* A cycle tunes the champion's enabled alphas, the disabled ones not marked failed and a fixed rotation of
  RESEARCH_FAILED_PER_CYCLE failed ones, so every alpha is still revisited regularly.
* A cycle is never due before twice the last cycle's length has passed (research duty cycle <= 1/2).
Synthetic data only."""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from heartless.config import RETENTION_SLACK_DAYS, Settings, research_retention_days
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.learning import lab, seal
from heartless.learning import optimizer as O
from heartless.learning.research import ResearchManager, describe_selection
from heartless.strategy import evidence
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_MINUTE

START = 1_704_067_200_000  # 2024-01-01 00:00 UTC (before the sealed 2025)
INFOS = {"AUSDT": {"symbol": "AUSDT", "base": "A", "quote": "USDT", "tick_size": 0.0001, "step_size": 0.001,
                   "min_qty": 0.001, "min_notional": 5.0, "price_precision": 4, "quantity_precision": 3}}


# --- the warm-up before the research window ----------------------------------------------------------------------------
@pytest.fixture
def long_alpha(monkeypatch):
    """trend_pullback declares a 40-day lookback and is replaced in the ensemble by a probe that records how many
    closed daily bars it sees at each decision (it never trades)."""
    monkeypatch.setattr(ALPHA_BY_NAME["trend_pullback"], "lookback_days", 40.0, raising=False)

    class Probe(Alpha):
        name = "trend_pullback"
        timeframe = "1h"
        seen: list = []

        def evaluate(self, view, ctx, p):
            Probe.seen.append((int(view.cursor_time), view.tf("1d").idx + 1))
            return None

    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Probe()], only_alpha))
    Probe.seen = []
    return Probe


def _walk_db(path, days: int, start: int = START):
    rng = np.random.default_rng(5)
    n = days * 1440
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    st = Store(path)
    rows, prev = [], float(close[0])
    for i in range(n):
        c, t = float(close[i]), start + i * MS_MINUTE
        rows.append(Candle(t, prev, max(prev, c) * 1.0003, min(prev, c) * 0.9997, c, 40.0, 40.0 * c, 4, 20.0, t + 59_999))
        prev = c
    st.save_candles("AUSDT", rows)
    st.close()
    return str(path)


def test_the_warmup_is_the_longest_lookback_of_the_tuned_alphas(long_alpha):
    p = StrategyParams.default()
    assert O.research_warmup_ms(p, ["htf_trend"]) == lab.WARMUP_MS  # no lookback declared: the lab's default
    assert O.research_warmup_ms(p, ["htf_trend", "trend_pullback"]) == 42 * MS_DAY  # 40 days + the 2-day margin
    assert O.research_warmup_ms(p, ["vol_breakout_dbox", "ts_trend_tsmom"]) == 92 * MS_DAY
    assert O.research_warmup_ms(p, []) == lab.WARMUP_MS
    assert O.longest_research_warmup_ms() == 92 * MS_DAY  # vol_breakout_dbox (90-day lookback)
    assert "discovered" not in O.research_alphas(p)  # improved by the discovery engine, not tuned here


def test_the_warmup_never_reaches_into_a_sealed_period(monkeypatch):
    a, b = 1_735_689_600_000, 1_767_225_600_000  # a synthetic seal (the real 2025 seal may be spent)
    monkeypatch.setattr(seal, "SEALED", {"2025": (a, b)})
    assert O.warmup_start(b + 10 * MS_DAY, 30 * MS_DAY) == b  # cut at the end of the seal
    assert O.warmup_start(b + 40 * MS_DAY, 30 * MS_DAY) == b + 10 * MS_DAY
    assert O.warmup_start(a - 5 * MS_DAY, 30 * MS_DAY) == a - 35 * MS_DAY
    assert O.warmup_start(b + 10 * MS_DAY, 30 * MS_DAY, unseal=["2025"]) == b - 20 * MS_DAY


def test_a_long_lookback_alpha_sees_its_whole_lookback_from_the_window_start(long_alpha, tmp_path):
    db = _walk_db(tmp_path / "r.db", 66)
    out = O.run_research_cycle(db, {"HEARTLESS_DATA_DIR": str(tmp_path)}, StrategyParams.default().to_dict(),
                               INFOS,
                               lookback_days=20, n_candidates=1, seed=1, alphas=["trend_pullback"], folds=1)
    assert "error" not in out, out
    w = out["window"]
    assert w["warmup_ms"] == 42 * MS_DAY and w["loaded_from"] == w["since"] - 42 * MS_DAY
    assert w["first_candle"] <= w["since"] - 41 * MS_DAY and out["selection"] == ["trend_pullback"]
    first_t, days = long_alpha.seen[0]
    assert first_t >= w["since"] and days >= 40  # the first research decision already sees 40+ closed days
    # before E2 the window was its own warm-up: the same first decision saw no closed daily bar at all
    long_alpha.seen = []
    orig = O.research_warmup_ms
    O.research_warmup_ms = lambda params, alphas: 0
    try:
        out0 = O.run_research_cycle(db, {"HEARTLESS_DATA_DIR": str(tmp_path)}, StrategyParams.default().to_dict(),
                                    INFOS,
                                    lookback_days=20, n_candidates=1, seed=1, alphas=["trend_pullback"], folds=1)
    finally:
        O.research_warmup_ms = orig
    assert out0["window"]["warmup_ms"] == 0 and long_alpha.seen[0][1] <= 1


def test_the_retention_covers_the_window_and_the_longest_warmup():
    warm_days = math.ceil(O.longest_research_warmup_ms() / MS_DAY)
    assert warm_days == 92 and research_retention_days(120) == 120 + 92 + RETENTION_SLACK_DAYS
    s = Settings(_env_file=None)
    assert s.data_retention_days == 214 > 180  # DATA_RETENTION_DAYS is a floor
    assert Settings(_env_file=None, DATA_RETENTION_DAYS=400).data_retention_days == 400
    assert Settings(_env_file=None, RESEARCH_LOOKBACK_DAYS=30).data_retention_days == 180  # 30 + 92 + 2 < 180
    assert Settings(_env_file=None, RESEARCH_LOOKBACK_DAYS=150).data_retention_days == 244
    # the worker rebuilds the settings from model_dump: the same value
    assert Settings(_env_file=None, **s.model_dump(by_alias=True)).data_retention_days == 214


# --- which alphas a cycle tunes ----------------------------------------------------------------------------------------
class _App:
    def __init__(self, tmp_path, **kw):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), TELEGRAM_BOT_TOKEN="", **kw)
        self.store = Store(tmp_path / "h.db")
        self.params = StrategyParams.default()
        self.universe, self.challengers, self.engines, self.symbols = [], [], {}, {}
        self.events = []

    async def emit(self, kind, payload):
        self.events.append((kind, payload))


def test_a_cycle_tunes_the_enabled_the_unjudged_and_a_rotation_of_the_failed(tmp_path, monkeypatch):
    app = _App(tmp_path)
    rm = ResearchManager(app)
    tunable = O.research_alphas(app.params)
    failed = sorted(a for a in tunable if evidence.verdict(a) == "failed")
    assert len(tunable) == 15 and failed == sorted(tunable)  # shipped: every parametrised alpha failed (P1, P2, E2)
    sel = rm.research_selection()
    assert sel == {"enabled": [], "untested": [], "failed": failed[:2], "failed_total": 15, "alphas": failed[:2]}
    assert rm.research_selection() == sel  # nothing advanced without advance=True
    # the champion's enabled alphas and an alpha the registry does not mark failed are tuned every cycle
    app.params.enabled["htf_trend"] = True
    real = evidence.verdict
    monkeypatch.setattr(evidence, "verdict", lambda a: "untested" if a == "xs_rank_mom" else real(a))
    sel = rm.research_selection()
    assert sel["enabled"] == ["htf_trend"] and sel["untested"] == ["xs_rank_mom"] and len(sel["failed"]) == 2
    assert "htf_trend" not in sel["failed"] and "xs_rank_mom" not in sel["failed"]
    assert sel["alphas"] == ["htf_trend", "xs_rank_mom"] + sel["failed"]
    assert describe_selection(sel).startswith("켜진 알파 htf_trend, 꺼진 미탈락 알파 xs_rank_mom, 탈락 알파 순환 2/13")


def test_the_failed_rotation_revisits_every_failed_alpha_and_survives_a_restart(tmp_path):
    app = _App(tmp_path, RESEARCH_FAILED_PER_CYCLE=2)
    rm = ResearchManager(app)
    failed = sorted(O.research_alphas(app.params))
    taken = []
    for _ in range(8):  # ceil(15 / 2) cycles
        taken += rm.research_selection(advance=True)["failed"]
    assert set(taken) == set(failed) and taken[:15] == failed and taken[15] == failed[0]  # wraps around
    nxt = rm.research_selection()["failed"]
    assert ResearchManager(app).research_selection()["failed"] == nxt  # a restart continues the rotation
    app.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), RESEARCH_FAILED_PER_CYCLE=0)
    assert ResearchManager(app).research_selection()["alphas"] == []


def test_the_cycle_hands_the_selection_to_the_worker_and_skips_an_empty_one(tmp_path, monkeypatch):
    app = _App(tmp_path)
    rm = ResearchManager(app)

    class _Pool:
        def shutdown(self, **k):
            pass

    async def run():
        loop = asyncio.get_running_loop()
        calls = []

        def submit(executor, fn, *args):
            calls.append(args)
            fut = loop.create_future()
            fut.set_result({"error": "no candles", "alphas": {}})
            return fut

        monkeypatch.setattr(loop, "run_in_executor", submit)
        rm.pool = _Pool()
        await rm.cycle(force=True)
        await rm.cycle(force=True)
        assert [c[7] for c in calls] == [sorted(O.research_alphas(app.params))[:2],
                                         sorted(O.research_alphas(app.params))[2:4]]
        started = [p["message"] for k, p in app.events if k == "research_started"]
        assert "탈락 알파 순환 2/15" in started[0]
        rm.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), RESEARCH_FAILED_PER_CYCLE=0)
        assert await rm.cycle(force=True) is None and len(calls) == 2 and not rm.running
        assert app.events[-1] == ("research_done", {"message": "리서치 건너뜀: 조정할 알파 없음", "notify": False})

    asyncio.run(run())


def test_the_research_duty_cycle_stays_at_most_one_half(tmp_path):
    app = _App(tmp_path, RESEARCH_INTERVAL_MINUTES=60)
    rm = ResearchManager(app)
    rm.last_cycle_ts = 1_000_000_000
    assert not rm.due(rm.last_cycle_ts + 59 * MS_MINUTE) and rm.due(rm.last_cycle_ts + 60 * MS_MINUTE)
    rm.last_cycle_ms = 50 * MS_MINUTE  # a cycle longer than half the interval (e.g. more enabled alphas)
    assert not rm.due(rm.last_cycle_ts + 99 * MS_MINUTE) and rm.due(rm.last_cycle_ts + 100 * MS_MINUTE)
    app.store.set("research.last_cycle_ms", 50 * MS_MINUTE)
    assert ResearchManager(app).last_cycle_ms == 50 * MS_MINUTE  # remembered across a restart


def test_the_default_interval_keeps_the_measured_cycle_under_half_of_it():
    s = Settings(_env_file=None)
    # docs/SELF_IMPROVEMENT.md 3.1: one shipped-default cycle (2 failed alphas, the 92-day warm-up pair) measured
    # E2_CYCLE_SECONDS on the research database copy
    assert E2_CYCLE_SECONDS * 2 <= s.research_interval_minutes * 60


E2_CYCLE_SECONDS = 1865  # ts_trend_tsmom + vol_breakout_dbox, research database copy (docs/SELF_IMPROVEMENT.md 3.1)

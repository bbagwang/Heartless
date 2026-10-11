"""P2a lab / backtester wiring: market context reaches alphas as ctx.market, universes, memory-bounded multi-year
pools and the trial ledger tag."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.learning import backtester as BT
from heartless.learning import lab, trials
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams

DAY = 86_400_000
INLINE = SimpleNamespace(map=map)
SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


def _ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


START = _ms(2023, 11, 1)


@pytest.fixture(autouse=True)
def _sandboxed_trial_ledger(tmp_path, monkeypatch):
    """Like the conftest seal-ledger fixture: no test may append to the real docs/results/p2_trials.jsonl."""
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "p2_trials.jsonl")
    return tmp_path / "p2_trials.jsonl"


class Probe(Alpha):
    """Records what it is given at every 1h close and never trades."""
    name = "trend_pullback"
    timeframe = "1h"
    seen: list = []

    def evaluate(self, view, ctx, p):
        Probe.seen.append((ctx.symbol, int(view.cursor_time), ctx.market))
        return None


@pytest.fixture
def probe(monkeypatch):
    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Probe()], only_alpha))
    Probe.seen = []
    return Probe


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("lab") / "h.db"
    st = Store(path)
    rng = np.random.default_rng(9)
    n = 70 * 1440  # 2023-11-01 .. 2024-01-09: crosses a year boundary
    for k, sym in enumerate(SYMS):
        close = (20 + 10 * k) * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
        prev = np.concatenate(([close[0]], close[:-1]))
        t = START + np.arange(n, dtype=np.int64) * 60_000
        st.save_candles(sym, [Candle(int(t[i]), float(prev[i]), float(max(prev[i], close[i]) * 1.0002),
                                     float(min(prev[i], close[i]) * 0.9998), float(close[i]), 10.0, 10.0 * float(close[i]),
                                     3, 5.0, int(t[i]) + 59_999) for i in range(n)])
    st.close()
    return str(path)


def _same(a, b) -> bool:
    return a.keys() == b.keys() and all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)


def _params():
    p = StrategyParams.default()
    p.enabled["trend_pullback"] = True
    return p


def test_lab_workers_hand_the_stored_context_to_alphas(db, tmp_path, probe):
    ctx_db = str(tmp_path / "ctx.db")
    a, b = START + 40 * DAY, START + 45 * DAY - 1
    res = lab.evaluate(_params(), ["AAAUSDT"], a, b, db, pool=INLINE, warmup_ms=12 * DAY,
                       context={"db": ctx_db, "universe": list(SYMS), "build": True})
    assert res.errors == {} and res.jobs[0]["context_hours"] > 100
    st = Store(ctx_db)
    table = C.ContextTable.load(st, ["AAAUSDT"], a - DAY, b)
    st.close()
    assert len(probe.seen) >= 5 * 24 - 2
    for sym, t, market in probe.seen:
        assert market["ts"] == C.hour_close(C.hour_of(t)) == t  # 1h alphas decide at the hour close itself
        assert _same(market, table.view(sym, t)) and market["n"] == 3.0
        with pytest.raises(TypeError):
            market["n"] = 0  # read-only


def test_without_context_alphas_see_an_empty_mapping_and_nothing_is_built(db, tmp_path, probe):
    ctx_db = tmp_path / "none.db"
    a, b = START + 40 * DAY, START + 41 * DAY - 1
    lab.evaluate(_params(), ["AAAUSDT"], a, b, db, pool=INLINE, warmup_ms=12 * DAY)
    lab.evaluate(_params(), ["AAAUSDT"], a, b, db, pool=INLINE, warmup_ms=12 * DAY,
                 context={"db": str(ctx_db), "universe": list(SYMS), "build": False})
    assert probe.seen and all(len(m) == 0 for _, _, m in probe.seen)
    assert not Store(ctx_db).has_market_context()


def test_backtester_and_load_backtester_attach_the_context(db, tmp_path, probe):
    ctx_db = tmp_path / "c.db"
    src, dst = Store(db), Store(ctx_db)
    C.build(src, dst, START + 30 * DAY, START + 50 * DAY, SYMS)
    dst.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), CONTEXT_DB=str(ctx_db))
    infos = {s: lab.infer_symbol_info(s, np.array([20.0, 20.01])) for s in SYMS}
    bt = BT.load_backtester(settings, src, infos, START + 30 * DAY, START + 42 * DAY, symbol_list=["BBBUSDT"])
    assert bt.market is not None and len(bt.market) > 200
    bt.run(_params(), START + 41 * DAY, START + 42 * DAY)
    assert probe.seen and all(m["ts"] == t and "ret_24h_rank" in m for _, t, m in probe.seen)
    off = BT.load_backtester(settings, src, infos, START + 30 * DAY, START + 42 * DAY, symbol_list=["BBBUSDT"],
                             market=None)
    assert off.market is None
    src.close()


def test_multi_year_runs_recycle_workers(db, monkeypatch):
    made = []

    class FakePool:
        def __init__(self, **kw):
            made.append(kw)

        def map(self, fn, jobs):
            return [{"symbol": j["symbol"], "chunk": j["label"], "start": j["start"], "end": j["end"], "trades": [],
                     "stats": {}, "error": "skipped"} for j in jobs]

        def shutdown(self):
            pass

    monkeypatch.setattr(lab, "ProcessPoolExecutor", FakePool)
    lab.evaluate(_params(), ["AAAUSDT"], START + 40 * DAY, START + 69 * DAY, db, workers=2)  # 2023 + 2024 chunks
    lab.evaluate(_params(), ["AAAUSDT"], START + 40 * DAY, START + 50 * DAY, db, workers=2)
    assert made[0]["max_tasks_per_child"] == 1 and made[1]["max_tasks_per_child"] is None


def test_universes():
    assert lab.universe_symbols("core") == list(C.CORE_SYMBOLS) and len(lab.universe_symbols("extended")) == 20
    with pytest.raises(ValueError):
        lab.universe_symbols("everything")


def _args(**kw):
    base = dict(symbols=None, universe=None, tag=None, years=None, start="2023-12-10", end="2023-12-13",
                split="train", unseal=[], params=None, set=[], stress=False, meta=False, chunk_days=None, alpha=None,
                workers=1, json=False, trades_out=None, out=None, by_side=False, argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_lab_cli_universe_and_tag_append_to_the_trial_ledger(db, tmp_path, monkeypatch, capsys, probe,
                                                             _sandboxed_trial_ledger):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), CONTEXT_DB=str(tmp_path / "ctx.db"))
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: db))
    real = lab.evaluate
    calls = []
    monkeypatch.setattr(lab, "evaluate", lambda *a, **k: calls.append((a, k)) or
                        real(*a, **{**k, "pool": INLINE, "warmup_ms": 12 * DAY}))
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(_params().to_dict()))
    with pytest.raises(SystemExit):
        lab.main_cli(settings, _args(symbols="AAAUSDT", universe="core"))
    lab.main_cli(settings, _args(universe="all", params=str(pfile), tag="probe_family"))
    assert calls[-1][0][1] == sorted(SYMS) and calls[-1][1]["context"]["universe"] == settings.context_universe_list
    lines = [json.loads(x) for x in _sandboxed_trial_ledger.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["tool"] == "lab" and lines[0]["family"] == "probe_family"
    assert lines[0]["n"] == 0 and lines[0]["count"] == 1 and lines[0]["period"] == "TRAIN"
    # the same configuration on another window or with stressed costs is not a new trial; a changed parameter is
    lab.main_cli(settings, _args(universe="all", params=str(pfile), tag="probe_family", start="2023-12-14",
                                 end="2023-12-15", stress=True))
    assert trials.family_tests("probe_family") == 1
    lab.main_cli(settings, _args(universe="all", params=str(pfile), tag="probe_family",
                                 set=["trend_pullback.adx_min=20"]))
    assert trials.family_tests("probe_family") == 2
    assert "family N=2" in capsys.readouterr().err


def test_context_for_a_window_right_after_the_seal_builds_only_what_it_reads(tmp_path, probe):
    """A VALID-style window starting 2026-01-01: the 2025 history is read as warm-up only; the one 2025 row written
    is the snapshot at 2025-12-31 23:59:59.999 that the window's first bars see."""
    from heartless.learning import seal

    s25_end = seal.SEALED["2025"][1]
    db = tmp_path / "s.db"
    st = Store(db)
    rng = np.random.default_rng(4)
    t0 = s25_end - 40 * DAY
    n = 50 * 1440
    for k, sym in enumerate(SYMS):
        close = (10 + k) * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
        st.save_candles(sym, [Candle(t0 + i * 60_000, float(close[i]), float(close[i]) * 1.0002,
                                     float(close[i]) * 0.9998, float(close[i]), 1.0, float(close[i]), 1, 0.5,
                                     t0 + i * 60_000 + 59_999) for i in range(n)])
    st.close()
    ctx_db = tmp_path / "ctx.db"
    with pytest.raises(seal.SealedError):
        lab.evaluate(_params(), ["AAAUSDT"], s25_end - DAY, s25_end + DAY, str(db), pool=INLINE,
                     context={"db": str(ctx_db), "universe": list(SYMS)})
    assert not ctx_db.exists()
    lab.evaluate(_params(), ["AAAUSDT"], s25_end, s25_end + 5 * DAY - 1, str(db), pool=INLINE, warmup_ms=12 * DAY,
                 context={"db": str(ctx_db), "universe": list(SYMS)})
    stats = Store(ctx_db).market_context_stats()
    assert stats["first"] == s25_end - 1 and stats["last"] >= s25_end + 5 * DAY - 1
    assert probe.seen and probe.seen[0][2]["ts"] >= s25_end - 1 and all(len(m) for _, _, m in probe.seen)

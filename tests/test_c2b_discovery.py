"""C2-B: the daily discovery mines every configured decision grid in ONE mine() call, so the multiple-testing N counts
the rules of all grids (it used to validate 15m and 1h separately, each against its own smaller N)."""
from __future__ import annotations

import asyncio

from heartless.config import Settings
from heartless.learning import discovery as D
from heartless.learning import research as R


def test_the_worker_mines_all_grids_in_one_call(monkeypatch):
    calls = []

    def mine(db_path, symbols, train, valid, tf="15m", cfg=None, workers=2, **kw):
        calls.append(tf)
        rules = [{"id": "a", "tf": "15m", "stats": {"valid": {"t": 1.6}}}, {"id": "b", "tf": "1h", "stats": {"valid": {"t": 2.4}}}]
        return {"tf": tf, "tested": 30_000, "n_tests": 30_000, "t_bar": D.multiple_testing_t(30_000), "results": [],
                "passed": rules, "seconds": 3.0, "rows": {"train": 10, "valid": 4}}

    monkeypatch.setattr(D, "mine", mine)
    out = R._discovery_worker("db", ["BTCUSDT"], (0, 1), (2, 3), "15m,1h")
    assert calls == ["15m,1h"]
    assert out["tested"] == out["n_tests"] == 30_000 and out["t_bar"] == D.multiple_testing_t(30_000)
    assert [r["id"] for r in out["passed"]] == ["b", "a"]  # strongest VALID t first, as before
    assert out["tf"] == "15m,1h" and "errors" not in out


def test_one_mine_call_uses_the_combined_multiple_testing_bar(monkeypatch):
    """The combined N is the sum of the rules evaluated on every grid: a stricter bar than either grid alone."""
    tested = {"15m": 20_000, "1h": 12_000}
    seen_bars = []

    def search(ds, cfg, progress=None):
        return [], tested[ds.tf]

    def validate(rules, ds, n_tests, *a, **k):
        seen_bars.append(n_tests)
        return []

    class _DS:
        def __init__(self, tf):
            self.tf, self.times = tf, [0]

    monkeypatch.setattr(D, "build_dataset", lambda db, syms, a, b, g, workers, **k: _DS(g))
    monkeypatch.setattr(D, "search", search)
    monkeypatch.setattr(D, "validate", validate)
    res = D.mine("db", ["BTCUSDT"], (1_772_323_200_000, 1_772_323_200_000 + 10), (1_773_000_000_000, 1_773_000_000_010),
                 "15m,1h", workers=1)
    assert res["n_tests"] == 32_000 and seen_bars == [32_000, 32_000]
    assert res["t_bar"] > D.multiple_testing_t(20_000)


def test_the_worker_reports_a_failed_run_without_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no data")

    monkeypatch.setattr(D, "mine", boom)
    out = R._discovery_worker("db", ["BTCUSDT"], (0, 1), (2, 3), "15m,1h")
    assert out["errors"] == ["15m,1h: no data"] and out["passed"] == [] and out["tested"] == 0


def test_the_cycle_hands_the_configured_grids_to_the_worker(tmp_path, monkeypatch):
    from heartless.core.store import Store
    from heartless.learning.research import ResearchManager
    from heartless.strategy.params import StrategyParams
    from heartless.util.timeutil import MS_DAY, now_ms

    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), DISCOVERY_GRIDS="15m,1h,1d", DISCOVERY_MIN_DAYS=20)
    assert Settings(_env_file=None).discovery_grids == "15m,1h"

    class _App:
        def __init__(self):
            self.s, self.store, self.params = s, Store(tmp_path / "h.db"), StrategyParams.default()
            self.universe, self.challengers, self.engines, self.events = ["BTCUSDT"], [], {}, []

        async def emit(self, kind, payload):
            self.events.append((kind, payload))

    app = _App()
    from heartless.core.models import Candle

    t0 = (now_ms() - 60 * MS_DAY) // MS_DAY * MS_DAY
    app.store.save_candles("BTCUSDT", [Candle(t0 + i * 3_600_000, 1, 1, 1, 1, 1, 1, 1, 1, t0 + i * 3_600_000 + 59_999)
                                       for i in range(60 * 24)])
    rm = ResearchManager(app)
    got = {}

    class _Loop:
        async def run_in_executor(self, pool, fn, *args):
            got["fn"], got["args"] = fn, args
            return {"tested": 0, "passed": [], "seconds": 0.0, "rows": {}, "t_bar": 0.0}

    monkeypatch.setattr(R, "_new_pool", lambda: object())
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())
    asyncio.run(rm.discovery_cycle(force=True))
    assert got["fn"] is R._discovery_worker and got["args"][-1] == "15m,1h,1d"

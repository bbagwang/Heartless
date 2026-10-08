"""Self-improvement loop: fold-confirmed parameter changes, re-enabling validated alphas, pruning losers,
discovery-driven challengers, and the rule-driven `discovered` alpha."""
import asyncio
from types import SimpleNamespace

import numpy as np

from heartless.config import Settings
from heartless.core.models import Regime, Side, Ticker
from heartless.core.store import Store
from heartless.learning import optimizer as O
from heartless.learning.research import ChallengerSlot, ResearchManager
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams


class _App:
    def __init__(self, tmp_path):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), CHALLENGERS=3)
        self.store = Store(tmp_path / "h.db")
        self.params = StrategyParams.default()
        self.universe, self.challengers, self.engines, self.symbols = [], [], {}, {}
        self.installed, self.events = [], []
        self.mode = "paper"

    async def emit(self, kind, payload):
        self.events.append((kind, payload))

    async def install_challenger(self, params, source):
        self.installed.append((source, params))
        return ChallengerSlot(name=f"challenger-{len(self.installed)}", params=params, started=0, source=source)


def _alpha_result(enabled, base_score, best_score, is_base=False, test_avg=0.1, test_n=30, folds_pos=3,
                  base_folds=None, k=3):
    return {"enabled": enabled, "n_folds": k,
            "base": {"params": {}, "train": {}, "test": {}, "score": base_score, "folds": base_folds or [], "folds_pos": 0},
            "best": {"params": {"min_conf": 0.6}, "train": {}, "test": {"avg_r": test_avg, "n": test_n}, "score": best_score,
                     "is_base": is_base, "distance": 0.1, "folds_pos": folds_pos}}


def test_apply_results_improves_reenables_and_prunes(tmp_path):
    app = _App(tmp_path)
    app.params.enabled["mean_reversion"] = False
    rm = ResearchManager(app)
    losing = [{"n": 20, "avg_r": -0.3}, {"n": 15, "avg_r": -0.2}, {"n": 12, "avg_r": -0.4}]
    result = {"alphas": {
        "trend_pullback": _alpha_result(True, 0.1, 1.0),  # better params for an enabled alpha
        "mean_reversion": _alpha_result(False, -1.0, 0.8, test_avg=0.12, test_n=25, folds_pos=2),  # comeback
        "momentum_burst": _alpha_result(True, -2.0, -1e9, is_base=True, base_folds=losing),  # consistent loser
        "squeeze_breakout": _alpha_result(True, 0.1, 1.0, folds_pos=1),  # one good fold only: rejected
    }}
    asyncio.run(rm._apply_results(result))
    by_src = {src: p for src, p in app.installed}
    assert set(by_src) == {"trend_pullback", "mean_reversion", "momentum_burst"}
    assert by_src["trend_pullback"].alphas["trend_pullback"]["min_conf"] == 0.6
    assert by_src["mean_reversion"].enabled["mean_reversion"] is True
    assert by_src["momentum_burst"].enabled["momentum_burst"] is False
    assert app.params.enabled["mean_reversion"] is False  # the champion itself is untouched


def test_confirm_folds_prefers_consistent_candidates():
    calls = []

    class BT:
        def run(self, params, a, b, only_alpha=None):
            v = params.alphas[only_alpha]["min_conf"]
            calls.append((v, a))
            # candidate A: great in one fold, terrible in others; candidate B: modestly positive everywhere
            avg = {0.5: {0: 0.9, 1: -0.4, 2: -0.4}, 0.6: {0: 0.15, 1: 0.12, 2: 0.1}}[v][a]
            return SimpleNamespace(stats={"n": 30, "avg_r": avg, "t_stat": avg * 5, "profit_factor": 1 + avg, "net": avg * 100,
                                          "max_dd": 10.0, "win_rate": 50.0, "expectancy": avg})

    base = StrategyParams.default()
    a = O.Candidate("trend_pullback", {**base.alphas["trend_pullback"], "min_conf": 0.5})
    b = O.Candidate("trend_pullback", {**base.alphas["trend_pullback"], "min_conf": 0.6})
    O.confirm_folds(BT(), base, "trend_pullback", [a, b], [(0, 1), (1, 2), (2, 3)])
    assert b.folds_pos == 3 and a.folds_pos == 1 and b.score > a.score


def test_discovery_cycle_installs_challenger_with_new_rules(tmp_path, monkeypatch):
    import heartless.learning.research as R

    app = _App(tmp_path)
    from heartless.core.models import Candle

    for sym in ("AAAUSDT", "BBBUSDT"):
        app.store.save_candles(sym, [Candle(t * 60_000, 1, 1, 1, 1, 1, 1, 1, 1, t * 60_000 + 59_999) for t in (0, 70 * 1440)])
    app.universe = ["AAAUSDT", "BBBUSDT"]
    rule = {"id": "rabc", "side": "LONG", "tf": "15m", "sl_atr": 1.5, "tp_r": 2.0, "hold_min": 480,
            "conds": [["15m.rsi14", "<", 30.0]], "stats": {"valid": {"n": 50, "avg_r": 0.2, "t": 2.1}}}
    monkeypatch.setattr(R, "_discovery_worker", lambda *a: {"tested": 1234, "t_bar": 3.8, "passed": [rule], "seconds": 1.0, "rows": {}})

    class _Pool:
        def shutdown(self, **k):
            pass

    async def run():
        rm = ResearchManager(app)
        rm.pool = _Pool()
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "run_in_executor", lambda pool, fn, *a: asyncio.sleep(0, result=fn(*a)))
        res = await rm.discovery_cycle(force=True)
        assert res["tested"] == 1234
        assert app.installed and app.installed[0][0] == "discovered"
        ch = app.installed[0][1]
        assert ch.enabled["discovered"] is True and ch.alphas["discovered"]["rules"][0]["id"] == "rabc"
        assert app.store.get("discovery.last")["tested"] == 1234

    asyncio.run(run())


def test_discovered_alpha_fires_only_when_its_rule_holds():
    from heartless.data.candles import CandleArrays, resample
    from heartless.data.features import FeatureFrame
    from heartless.core.models import Candle, SymbolInfo

    rng = np.random.default_rng(5)
    ca = CandleArrays("1m", 30 * 1440 + 16)
    px = 100.0
    for i in range(30 * 1440):
        o = px
        px *= 1 + rng.normal(0, 0.001)
        ca.append(Candle(i * 60_000, o, max(o, px) * 1.0005, min(o, px) * 0.9995, px, 10, 10 * px, 5, 5, i * 60_000 + 59_999))
    frames = {tf: FeatureFrame(tf, resample(ca, tf)) for tf in ("15m", "1h")}
    t = int(frames["15m"].close_time[-1])
    view = SimpleNamespace(frames=frames, cursor_time=t, closed=lambda tf: True,
                           tf=lambda tf: SimpleNamespace(v=lambda name, k=0: float(frames[tf].f[name][-1 - k])))
    info = SymbolInfo("X", "X", "USDT", 0.01, 0.001, 0.001, 5, 2, 3)
    ctx = Context("X", info, Ticker("X", bid=px, ask=px, mark=px), Regime.RANGE)
    rsi = float(frames["15m"].f["rsi14"][-1])
    alpha = ALPHA_BY_NAME["discovered"]
    base_rule = {"id": "r1", "side": "SHORT", "tf": "15m", "sl_atr": 1.5, "tp_r": 2.0, "hold_min": 480}
    hit = alpha.evaluate(view, ctx, {"min_conf": 0.55, "rules": [{**base_rule, "conds": [["15m.rsi14", "<", rsi + 1]]}]})
    miss = alpha.evaluate(view, ctx, {"min_conf": 0.55, "rules": [{**base_rule, "id": "r2", "conds": [["15m.rsi14", "<", rsi - 1]]}]})
    assert miss is None and hit is not None and hit.side is Side.SHORT
    assert hit.stop > hit.tags["ref_price"] > hit.take_profit  # short geometry
    assert alpha.evaluate(view, ctx, {"min_conf": 0.55}) is None  # no rules -> silent


def test_run_research_cycle_end_to_end_with_folds(tmp_path):
    from dataclasses import asdict

    from synth import synth_candles, synth_symbols

    st = Store(tmp_path / "r.db")
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT")):
        ca = synth_candles(14 * 1440, seed=i, price=100 * (i + 1))
        st.save_candles(sym, [ca.candle_at(k) for k in range(ca.n)])
    st.close()
    infos = {k: asdict(v) for k, v in synth_symbols(["BTCUSDT", "ETHUSDT"]).items()}
    out = O.run_research_cycle(str(tmp_path / "r.db"), {}, StrategyParams.default().to_dict(), infos, lookback_days=10,
                               n_candidates=3, seed=1, alphas=["trend_pullback", "discovered"], folds=3)
    assert "error" not in out, out
    assert "discovered" not in out["alphas"]  # rule-driven alpha is left to the discovery engine
    tp = out["alphas"]["trend_pullback"]
    assert tp["n_folds"] == 3 and len(out["window"]["folds"]) == 3 and "folds" in tp["best"]

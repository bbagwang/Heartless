"""C2-A discovered alpha: every rule type the miner produces runs, on features equal to the miner's bit for bit
(frame, positioning, time and market-context families) on the 15m, 1h and 1d grids."""
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle, Regime, Side, SymbolInfo, Ticker
from heartless.core.store import Store
from heartless.data import context as C
from heartless.data.candles import CandleArrays
from heartless.data.extras import MetricsSeries
from heartless.data.features import MarketView
from heartless.learning import discovery as D
from heartless.learning.backtester import FUNDING_INTERVAL, Backtester
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.discovered import Discovered
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

H = 3_600_000
DAY = 86_400_000
START = 1_701_734_400_000  # 2023-12-05 00:00 UTC
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DAYS = 60
WINDOW = (START + 40 * DAY, START + 46 * DAY - 1)  # crosses no year boundary; 22-day warm-up inside the history


def _metrics(k: int, rng) -> list:
    """5m rows with the awkward cases the snapshot handles: zero / missing ratios, a stale gap, an OI dip to zero."""
    rows = []
    oi = 1e6 * (1 + k)
    for j in range(DAYS * 288):
        if 4000 + 300 * k <= j < 4012 + 300 * k:
            continue  # an hour without rows: snapshots go stale
        oi *= 1 + rng.normal(0, 0.002)
        tk = float(abs(rng.normal(1.0, 0.2)))
        rows.append(SimpleNamespace(
            ts=START + j * 300_000, oi=0.0 if j % 997 == 0 else oi, oi_value=oi * 20,
            top_ls_accounts=1.1, top_ls_positions=None if j % 401 == 0 else float(abs(rng.normal(1.2, 0.1))),
            ls_accounts=float(abs(rng.normal(1.0, 0.05))), taker_ls_vol=0.0 if j % 13 == 0 else tk))
    return rows


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Candles, funding and metrics for four symbols, and the context built from them."""
    d = tmp_path_factory.mktemp("disc")
    st = Store(d / "heartless.db")
    rng = np.random.default_rng(11)
    common = rng.normal(0, 0.0007, DAYS * 1440)
    for k, sym in enumerate(SYMS):
        close = (50 + 25 * k) * np.exp(np.cumsum(0.7 * common + rng.normal(0, 0.0007, DAYS * 1440)))
        bars, prev = [], float(close[0])
        for i in range(DAYS * 1440):
            c = float(close[i])
            v = float(abs(rng.normal(40, 12)) + 1)
            t = START + i * 60_000
            bars.append(Candle(t, prev, max(prev, c) * 1.0004, min(prev, c) * 0.9996, c, v, v * c, 5,
                               v * float(rng.uniform(0.3, 0.7)), t + 59_999))
            prev = c
        st.save_candles(sym, bars)
        st.save_funding(sym, [(START + j * 8 * H, float(rng.normal(1e-4, 2e-4)), 0.0) for j in range(DAYS * 3)])
        st.save_metrics(sym, _metrics(k, rng))
    ctx = Store(d / "context.db")
    C.build(st, ctx, universe=SYMS)
    ctx.close()
    st.close()
    return {"db": str(d / "heartless.db"), "ctx": str(d / "context.db")}


def _job(world, sym: str, grid: str) -> dict:
    return {"db_path": world["db"], "symbol": sym, "start": WINDOW[0], "end": WINDOW[1], "tf": grid,
            "templates": (1, 4), "context_db": world["ctx"]}


def _inputs(world, sym: str):
    """Exactly what discovery._build_job loads for the window (1m history with its warm-up, metrics, context)."""
    st = Store(world["db"])
    hold = max(D.TEMPLATES[t][2] for t in (1, 4))
    rows = st.load_candles(sym, start=WINDOW[0] - D.WARMUP_MS, end=WINDOW[1] + hold * 60_000 + H)
    ms = MetricsSeries.from_rows(st.load_metrics(sym, WINDOW[0] - 4 * DAY, WINDOW[1]))
    funding = st.load_funding(sym, WINDOW[0] - D.WARMUP_MS - FUNDING_INTERVAL, WINDOW[1] + FUNDING_INTERVAL)
    st.close()
    ca = CandleArrays("1m", capacity=len(rows) + 16)
    ca.extend(rows)
    cst = Store(world["ctx"])
    market = C.ContextTable.load(cst, [sym], WINDOW[0] - C.MS_H, WINDOW[1], source=world["db"])
    cst.close()
    return ca, ms, funding, market


def _bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per element: equal bits, or both NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (np.isnan(a) & np.isnan(b)) | (a.view(np.int64) == b.view(np.int64))


NEVER = [["t.hour_utc", ">", 99.0]]  # rules that never fire: no position ever blocks an evaluation


def test_the_alpha_sees_the_miners_feature_vector_on_every_grid_and_family(world, monkeypatch):
    sym = "ETHUSDT"
    ca, ms, funding, market = _inputs(world, sym)
    seen: dict[int, np.ndarray] = {}
    orig = Discovered.features

    def record(view, ctx, names=None):
        x = orig(view, ctx, names)
        seen[int(view.cursor_time)] = x
        return x

    monkeypatch.setattr(Discovered, "features", staticmethod(record))
    rules = [{"id": f"never-{g}", "side": "LONG", "tf": g, "sl_atr": 4.0, "tp_r": None, "hold_min": 1440,
              "conds": NEVER} for g in ("15m", "1h", "1d")]
    p = StrategyParams.default()
    p.enabled["discovered"] = True
    p.alphas["discovered"]["rules"] = rules
    info = SymbolInfo(sym, sym[:-4], "USDT", 0.001, 0.001, 0.001, 5.0, 3, 3)
    bt = Backtester(Settings(_env_file=None), {sym: info}, {sym: ca}, {sym: funding}, {sym: ms}, market)
    bt.run(p, WINDOW[0], WINDOW[1], only_alpha="discovered")
    names = D.FEATURES_X
    fams = {"frame": [i for i, n in enumerate(names) if n[:3] in ("15m", "1h.")],
            "positioning": [i for i, n in enumerate(names) if n.startswith("x.")],
            "time": [i for i, n in enumerate(names) if n.startswith("t.")],
            "context": [i for i, n in enumerate(names) if n.startswith(("c.", "m."))]}
    for grid, n_rows in (("15m", 6 * 96), ("1h", 6 * 24), ("1d", 6)):
        part = D._build_job(_job(world, sym, grid))
        assert part.names == names and len(part.times) == n_rows
        for j, t in enumerate(part.times.tolist()):
            assert t in seen, (grid, C._iso(t))  # the alpha evaluated at every decision time of the grid
            ok = _bits(seen[t], part.X[j])
            assert ok.all(), (grid, C._iso(t), [names[i] for i in np.flatnonzero(~ok)])
        for fam, cols in fams.items():  # none of the comparisons is vacuous
            assert np.isfinite(part.X[:, cols]).any(axis=0).mean() > 0.6, (grid, fam)
    # the positioning family covers stale stretches (all NaN) and live ones
    xi = names.index("x.oi_chg_1h_z")
    vals = np.array([seen[t][xi] for t in sorted(seen)])
    assert np.isfinite(vals).sum() > 100 and np.isnan(vals).sum() > 0


def test_positioning_features_equal_the_snapshot_bit_for_bit(world):
    """extras_features() (the miner) and MetricsSeries.snapshot() (ctx.extras in the bot and the lab) agree exactly,
    including the 1h taker geometric mean and the OI-change z-score."""
    st = Store(world["db"])
    ms = MetricsSeries.from_rows(st.load_metrics("BTCUSDT"))
    st.close()
    times = np.arange(START + 3 * DAY, START + 30 * DAY, 7 * 60_000 + 13, dtype=np.int64)
    batch = D.extras_features(ms, times)
    for j, t in enumerate(times.tolist()):
        snap = ms.snapshot(t)
        live = {} if snap.get("stale") else snap
        for n in D._EXTRAS:
            v = live.get(n)
            ref = float(v) if isinstance(v, float) else np.nan
            assert _bits(np.array([ref]), batch[f"x.{n}"][j:j + 1]).all(), (n, t, ref, batch[f"x.{n}"][j])


def _view(world, sym: str, t: int):
    ca, ms, _, market = _inputs(world, sym)
    v = MarketView(sym)
    v.rebuild(ca, live=False)
    v.seek(t)
    info = SymbolInfo(sym, sym[:-4], "USDT", 0.001, 0.001, 0.001, 5.0, 3, 3)
    px = v.price
    ctx = Context(sym, info, Ticker(sym, bid=px, ask=px, mark=px), Regime.RANGE, now=t, extras=ms.snapshot(t),
                  market=market.view(sym, t))
    return v, ctx


def test_time_exit_rules_on_the_daily_grid_decide_only_at_midnight_utc(world):
    alpha = ALPHA_BY_NAME["discovered"]
    rule = {"id": "rd", "side": "SHORT", "tf": "1d", "sl_atr": 4.0, "tp_r": None, "hold_min": 4320,
            "conds": [["t.hour_utc", ">", -1.0]]}
    midnight = WINDOW[0] + 2 * DAY - 1
    v, ctx = _view(world, "BTCUSDT", midnight)
    sig = alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [rule]})
    atr = v.tf("1h").v("atr")
    assert sig is not None and sig.side is Side.SHORT and sig.take_profit is None and sig.max_hold_bars == 4320
    assert sig.stop == pytest.approx(v.tf("1h").v("close") + 4.0 * atr) and sig.timeframe == "1h"
    assert sig.tags["time_exit"] and sig.tags["grid"] == "1d" and "시간 청산 72시간" in sig.reason
    for t in (midnight - H, midnight + H, midnight + 15 * 60_000):  # other hour / quarter closes of the day
        v.seek(t)
        ctx.now = t
        assert alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [rule]}) is None
    # a classic rule keeps its target
    classic = {**rule, "id": "rc", "tf": "1h", "sl_atr": 1.5, "tp_r": 2.0, "hold_min": 480}
    v.seek(midnight - H)
    sig = alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [classic]})
    dist = 1.5 * v.tf("1h").v("atr")
    assert sig.take_profit == pytest.approx(sig.tags["ref_price"] - 2.0 * dist) and not sig.tags["time_exit"]


def test_context_rules_read_ctx_market_and_unknown_rules_are_ignored(world):
    alpha = ALPHA_BY_NAME["discovered"]
    t = WINDOW[0] + 3 * DAY + 6 * H - 1
    v, ctx = _view(world, "SOLUSDT", t)
    rank = ctx.market["ret_24h_rank"]
    breadth = ctx.market["breadth_ema50"]
    assert rank == rank and breadth == breadth
    base = {"side": "LONG", "tf": "1h", "sl_atr": 1.5, "tp_r": 2.0, "hold_min": 480}
    hit = {**base, "id": "c1",
           "conds": [["c.ret_24h_rank", "<", rank + 1e-9], ["m.breadth_ema50", ">", breadth - 1e-9]]}
    miss = {**base, "id": "c2", "conds": [["c.ret_24h_rank", ">", rank + 1e-9]]}
    assert alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [hit]}) is not None
    assert alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [miss]}) is None
    ctx.market = C.EMPTY  # no context available: the condition is unknown, the rule does not fire
    assert alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": [hit]}) is None
    unknown = [{**base, "id": "u1", "conds": [["z.nope", "<", 1.0]]}, {**base, "id": "u2", "tf": "3h", "conds": NEVER}]
    assert alpha.evaluate(v, ctx, {"min_conf": 0.55, "rules": unknown}) is None
    assert Discovered.lookback_days == D.WARMUP_MS / DAY


def test_a_daily_time_exit_rule_trades_through_the_engine(world):
    sym = "XRPUSDT"
    ca, ms, funding, market = _inputs(world, sym)
    p = StrategyParams.default()
    p.enabled["discovered"] = True
    p.alphas["discovered"]["rules"] = [{"id": "rt", "side": "LONG", "tf": "1d", "sl_atr": 4.0, "tp_r": None,
                                        "hold_min": 480, "conds": [["t.hour_utc", ">", -1.0]]}]
    info = SymbolInfo(sym, sym[:-4], "USDT", 0.0001, 0.1, 0.1, 5.0, 4, 1)
    bt = Backtester(Settings(_env_file=None), {sym: info}, {sym: ca}, {sym: funding}, {sym: ms}, market)
    res = bt.run(p, WINDOW[0], WINDOW[1], only_alpha="discovered")
    assert len(res.trades) >= 4
    for tr in res.trades:
        assert tr["entry_time"] % DAY == 0  # filled at the open of the 00:00 UTC bar after the daily decision
        if tr["exit_reason"].startswith("시간 초과"):
            assert tr["exit_time"] == tr["entry_time"] + 480 * 60_000 - 1  # at the close of the 480th minute held

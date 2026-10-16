"""Design-agnostic invariants for every registered alpha, on a real MarketView over synthetic 1m data.

Replaces the per-design regression tests of the retired 5m trend_pullback / mean_reversion logic (NaN EMAs read as a
trend, a take-profit emitted on the wrong side of entry). Whatever an alpha's setup is, every Signal it emits must be a
bracket the engine can place: stop on the protective side of the reference price, take-profit and partial on the
favourable side (partial inside the final target), a finite positive ATR, a 1m time stop and a ref_price tag. The data
starts cold, so the warm-up NaN path of every alpha is exercised as well.

Two drives: 21 days of 1m bars evaluated at every 15m close (the intraday alphas), and E1: 130 days with multi-day
volatility / drift regimes evaluated at every 1h close, i.e. across every 4h and UTC day close, with a synthetic market
context and positioning snapshot, so the 4h / 1d alphas (positioning_flow_oimom, vol_breakout_dbox, ts_trend_tsmom,
xs_rank_mom) and the context-driven 1h ones decide on real feature frames and must emit valid brackets, only on their
own timeframe's close.

F0 (symbol-agnostic): every alpha must decide the same way whatever the symbol is called and whatever its price
level is. The same drives are repeated with the symbol renamed to a random ticker and with every price scaled by 1000
and by 0.001 (tick size scaled with the price, step size and base volume inversely, quote volume unchanged); the
decisions must match one for one: same bars, sides, entry styles, timeframes and holds, prices in proportion (relative
1e-9) and confidences equal (relative 1e-9). A scale of 1024 (exact in binary floating point) separates a real price-level
dependence from rounding noise near a threshold.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest
from synth import synth_candles, synth_symbols

from heartless.core.models import EntryStyle, Regime, Side, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.strategy.alphas import ALL_ALPHAS
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR

SYM = "ETHUSDT"
DAYS = 21
LONG_DAYS = 130  # > vol_breakout_dbox.lookback_days (90): every long-lookback alpha gets past its warm-up
START = 1_700_006_400_000  # 00:00 UTC
SLOW_TFS = ("4h", "1d")
# parameter sets evaluated besides the shipped defaults: the researched configurations (docs/RESEARCH_P2.md), so an alpha
# whose default filter is rare on synthetic data still emits enough brackets to check
PARAM_VARIANTS = {"vol_breakout_dbox": ({"comp_q": 1.0},)}


@pytest.fixture(scope="module")
def signals():
    params = StrategyParams.default()
    info = synth_symbols([SYM])[SYM]
    out: dict[str, list] = {a.name: [] for a in ALL_ALPHAS}
    # a calm and a very volatile series (some alphas only trade when the 1h ATR is large)
    for seed, vol in ((3, 0.0008), (5, 0.0022)):
        ca = synth_candles(DAYS * 1440, seed=seed, price=100.0, vol=vol)
        view = MarketView(SYM)
        view.rebuild(ca, live=False)
        close_t, close = ca.view("close_time"), ca.view("close")
        for i in range(0, ca.n, 15):  # every 15m close (no registered alpha decides on a faster timeframe)
            j = min(i + 14, ca.n - 1)
            t = int(close_t[j])
            view.seek(t)
            px = float(close[j])
            tk = Ticker(SYM, bid=px * 0.99995, ask=px * 1.00005, mark=px, last=px, ts=t)
            for regime in (Regime.RANGE, Regime.TREND_UP):
                ctx = Context(symbol=SYM, info=info, ticker=tk, regime=regime, now=t, extras={})
                for a in ALL_ALPHAS:
                    if not view.closed(a.timeframe):
                        continue
                    s = a.evaluate(view, ctx, params.alphas[a.name])
                    if s is not None:
                        out[a.name].append((s, px))
    return out


def long_candles(days: int = LONG_DAYS, seed: int = 21, price: float = 100.0) -> CandleArrays:
    """1m bars whose volatility and drift switch every 3-12 days (calm / normal / wild, down / flat / up), so the daily
    frames see compressions, box breakouts and multi-week trends; vectorised (seconds for 187k bars)."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    vol, drift = np.empty(n), np.empty(n)
    i = 0
    while i < n:
        span = int(rng.integers(3, 13)) * 1440
        vol[i:i + span] = rng.choice([0.0004, 0.0009, 0.0018], p=[0.35, 0.4, 0.25])
        drift[i:i + span] = rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.00002, 0.00008)
        i += span
    ret = drift + rng.standard_t(4, n) * vol * 0.6
    close = price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[price], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1, n)) * vol * 0.5)
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1, n)) * vol * 0.5)
    volume = np.abs(rng.normal(50, 20, n)) * (1 + np.abs(ret) / vol) + 1
    ca = CandleArrays("1m", capacity=n + 16)
    ot = START + np.arange(n, dtype=np.int64) * 60_000
    cols = {"open_time": ot, "close_time": ot + 59_999, "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": volume * close, "trades": np.round(volume),
            "taker_buy": volume * np.clip(0.5 + ret / (vol * 3), 0.05, 0.95)}
    for name, arr in cols.items():
        getattr(ca, name)[:n] = arr
    ca.n = n
    return ca


def market(now: int, up: bool) -> dict:
    """A context snapshot leaning one way: BTC's week, the symbol's return ranks, its funding against the median."""
    sg = 1.0 if up else -1.0
    return {"ts": now, "n": 20.0, "btc_ret_168h": 0.06 * sg, "ret_24h_rank": 0.5 + 0.45 * sg,
            "ret_72h_rank": 0.5 + 0.45 * sg, "ret_168h_rank": 0.5 + 0.45 * sg, "vol_share_rank": 0.8,
            "funding": 0.0001 - 0.0006 * sg, "funding_median": 0.0001, "funding_rank": 0.5 - 0.45 * sg,
            "oi_chg_24h": 0.05, "ret_24h": -0.08 if not up else 0.03, "disp_24h": 0.05}


def extras(up: bool) -> dict:
    """A positioning snapshot: open interest building, retail accounts leaning against `up`."""
    return {"oi_chg_24h": 0.15, "oi_chg_4h": 0.03, "ls_acc_chg_4h": -0.02 if up else 0.02, "top_ls_pos": 1.2,
            "top_ls_pos_chg_4h": 0.01, "oi_chg_1h_z": 1.0, "age_min": 1.0}


@pytest.fixture(scope="module")
def long_run():
    """Every alpha at every 1h close of LONG_DAYS (so across all 4h and day closes), both context leanings."""
    params = StrategyParams.default()
    info = synth_symbols([SYM])[SYM]
    ca = long_candles()
    view = MarketView(SYM)
    view.rebuild(ca, live=False)
    close_t, close = ca.view("close_time"), ca.view("close")
    out: dict[str, list] = defaultdict(list)
    asked: dict[str, set] = defaultdict(set)  # alpha -> timeframes whose close it was asked at
    for j in range(59, ca.n, 60):
        t = int(close_t[j])
        view.seek(t)
        px = float(close[j])
        tk = Ticker(SYM, bid=px * 0.99995, ask=px * 1.00005, mark=px, last=px, ts=t)
        closed = {tf for tf in ("1h", *SLOW_TFS) if view.closed(tf)}
        assert ("4h" in closed) == ((t + 1) % (4 * MS_HOUR) == 0) and ("1d" in closed) == ((t + 1) % MS_DAY == 0)
        for up in (True, False):
            ctx = Context(symbol=SYM, info=info, ticker=tk, regime=Regime.RANGE, now=t, extras=extras(up),
                          market=market(t, up), funding_rate=0.0001, minutes_to_funding=200.0)
            for a in ALL_ALPHAS:
                if not view.closed(a.timeframe):
                    continue
                asked[a.name] |= closed
                for over in ({}, *PARAM_VARIANTS.get(a.name, ())):
                    s = a.evaluate(view, ctx, {**params.alphas[a.name], **over})
                    if s is not None:
                        out[a.name].append((s, px, frozenset(closed)))
    return out, asked


def test_every_alpha_is_driven_across_4h_and_day_closes(long_run):
    _, asked = long_run
    for a in ALL_ALPHAS:
        assert {"4h", "1d"} <= asked[a.name], (a.name, asked[a.name])


def test_slow_alphas_fire_and_only_on_their_own_close(long_run):
    signals, _ = long_run
    slow = [a for a in ALL_ALPHAS if a.timeframe in SLOW_TFS]
    assert {a.name for a in slow} >= {"positioning_flow_oimom", "vol_breakout_dbox", "ts_trend_tsmom", "xs_rank_mom"}
    for a in slow:
        assert signals[a.name], f"{a.name} never fired on {LONG_DAYS} days of 4h / 1d closes"
        assert all(a.timeframe in closed for _, _, closed in signals[a.name]), a.name


def test_long_run_signals_are_valid_brackets(long_run):
    signals, _ = long_run
    check_brackets({k: [(s, px) for s, px, _ in v] for k, v in signals.items()})


def test_enough_alphas_fire_for_the_invariants_to_mean_something(signals):
    fired = {a for a, v in signals.items() if v}
    assert len(fired) >= 4, {a: len(v) for a, v in signals.items()}


def test_every_signal_is_a_valid_bracket(signals):
    check_brackets(signals)


def check_brackets(signals: dict) -> None:
    for name, items in signals.items():
        for s, px in items:
            ref = s.tags.get("ref_price")
            assert s.alpha == name and s.symbol == SYM
            assert ref is not None and math.isfinite(ref) and ref > 0, (name, s.tags)
            assert abs(ref / px - 1) < 0.01, (name, ref, px)  # decided on the just-closed bar, not elsewhere
            sg = s.side.sign
            assert (ref - s.stop) * sg > 0, (name, s.side, ref, s.stop)  # protective stop
            if s.take_profit is not None:
                assert (s.take_profit - ref) * sg > 0, (name, s.side, ref, s.take_profit)
            if s.tp1 is not None:
                assert (s.tp1 - ref) * sg > 0, (name, ref, s.tp1)
                if s.take_profit is not None:
                    assert (s.take_profit - s.tp1) * sg > 0, (name, s.tp1, s.take_profit)
            if s.entry_style is EntryStyle.LIMIT:
                lp = s.limit_price
                assert lp is not None and lp > 0 and (lp - s.stop) * sg > 0, (name, lp, s.stop)
                if s.take_profit is not None:
                    assert (s.take_profit - lp) * sg > 0, (name, lp, s.take_profit)
            assert 0.0 < s.confidence <= 1.0 and s.reason
            assert math.isfinite(s.atr) and s.atr > 0
            assert s.max_hold_bars > 0 and s.timeframe in ("1m", "5m", "15m", "1h", "4h", "1d")
            assert s.side in (Side.LONG, Side.SHORT)


def test_no_emitted_signal_sets_an_execution_opt_in(signals, long_run):
    """E1 opt-ins stay off for every shipped alpha (their evidence was measured without them), checked on every signal
    the two drives produced rather than on the source text, which an assignment such as `sig.time_exit = True` or a
    keyword passed through a dict would slip past."""
    from heartless.core.models import Signal

    default = Signal(alpha="x", symbol=SYM, side=Side.LONG, confidence=0.5, reason="x", stop=1.0, take_profit=None,
                     tp1=None)
    keys = ("time_exit", "allow_renewal", "exit_on_opposite", "cooldown_bars", "avoid_funding")
    emitted = [s for v in signals.values() for s, _ in v] + [s for v in long_run[0].values() for s, _, _ in v]
    assert emitted
    for s in emitted:
        assert tuple(getattr(s, k) for k in keys) == tuple(getattr(default, k) for k in keys), (s.alpha, s)


# --- F0: symbol-agnostic invariants -----------------------------------------------------------------------------------------
RENAMED = "QZJXWKUSDT"  # a ticker no alpha could know
AGNOSTIC_SCALES = (1000.0, 0.001, 1024.0)
REL = 1e-9
# permissive settings for the alphas whose shipped filters never trigger on the synthetic drives: the invariance is a
# property of the code, so it is checked where the code actually decides
AGNOSTIC_VARIANTS = {"momentum_burst": ({"burst_atr": 1.0, "vol_z_min": 1.0, "trend_align": 0},),
                     "funding_fade": ({"settle_win": 0, "z_min": 1.0, "fz_min": -1.0},),
                     # discovered fires only with mined rules: three on scale-free features, one per decision grid
                     "discovered": ({"rules": [
                         {"id": "f0a", "side": "LONG", "tf": "1h", "sl_atr": 2.0, "tp_r": 2.0, "hold_min": 240,
                          "conds": [["1h.rsi14", ">", 60.0], ["1h.dist_ema50", ">", 0.0]]},
                         {"id": "f0b", "side": "SHORT", "tf": "15m", "sl_atr": 2.0, "tp_r": 2.0, "hold_min": 120,
                          "conds": [["15m.bb_z", "<", -1.5]]},
                         {"id": "f0c", "side": "LONG", "tf": "1d", "sl_atr": 4.0, "tp_r": None, "hold_min": 1440,
                          "conds": [["1h.ret4_atr", ">", 0.0]]}]},)}
# F0 finding (docs/UNIVERSE.md 5): vwap_z = (close - session VWAP) / session sd, with sd = sqrt(E[p^2 v] / E[v] - vwap^2).
# On the first bar of a UTC session the true sd is 0 and the computed one is rounding noise, so the feature is +-1e5 or
# NaN depending on the price level's rounding. Differences of these features are accepted on that bar only.
NUMERIC_FEATURES = {"15m.vwap_z", "1h.vwap_z"}
FEATURE_RTOL, FEATURE_ATOL = 1e-7, 1e-9  # rolling variances (bb_z) differ by ~1e-11 between price levels


def moving_extras(t: int, up: bool) -> tuple[dict, float, float]:
    """extras / funding rate / minutes to funding that move with time (deterministic in t, free of any price level):
    the top-trader ratio and the funding rate swing over hours, so z-score based alphas see extremes."""
    ex = dict(extras(up))
    ex["top_ls_pos"] = 1.2 * math.exp(0.35 * math.sin(t / (9 * MS_HOUR)) + 0.1 * math.sin(t / (2.3 * MS_HOUR)))
    fr = 0.0001 + 0.0005 * math.sin(t / (13 * MS_HOUR) + 0.7)
    return ex, fr, float(480 - (t // 60_000) % 480)


def scaled_candles(ca: CandleArrays, scale: float) -> CandleArrays:
    """The same market quoted `scale` times higher: prices x scale, base quantities / scale, notional unchanged."""
    out = CandleArrays("1m", capacity=ca.n + 16)
    for name in ("open_time", "close_time", "quote_volume", "trades"):
        getattr(out, name)[:ca.n] = getattr(ca, name)[:ca.n]
    for name in ("open", "high", "low", "close"):
        getattr(out, name)[:ca.n] = getattr(ca, name)[:ca.n] * scale
    for name in ("volume", "taker_buy"):
        getattr(out, name)[:ca.n] = getattr(ca, name)[:ca.n] / scale
    out.n = ca.n
    return out


def scaled_info(symbol: str, scale: float):
    from dataclasses import replace

    base = synth_symbols([SYM])[SYM]
    return replace(base, symbol=symbol, base=symbol[:-4], tick_size=base.tick_size * scale,
                   step_size=base.step_size / scale, min_qty=base.min_qty / scale)


def decisions(ca: CandleArrays, symbol: str, scale: float, every: int = 60) -> dict[str, list[tuple]]:
    """Every signal of every alpha (shipped defaults and PARAM_VARIANTS, both context leanings) at every `every`-minute
    close, as (time, leaning, variant, side, style, timeframe, hold, confidence, prices / scale...)."""
    params = StrategyParams.default()
    alphas = [type(a)() for a in ALL_ALPHAS]  # fresh instances: some keep per-symbol history between calls
    info = scaled_info(symbol, scale)
    view = MarketView(symbol)
    view.rebuild(ca, live=False)
    close_t, close = ca.view("close_time"), ca.view("close")
    out: dict[str, list[tuple]] = defaultdict(list)
    for j in range(every - 1, ca.n, every):
        t = int(close_t[j])
        view.seek(t)
        px = float(close[j])
        tk = Ticker(symbol, bid=px * 0.99995, ask=px * 1.00005, mark=px, last=px, ts=t)
        for up in (True, False):
            ex, fr, mtf = moving_extras(t, up)
            ctx = Context(symbol=symbol, info=info, ticker=tk, regime=Regime.RANGE, now=t, extras=ex,
                          market=market(t, up), funding_rate=fr, minutes_to_funding=mtf)
            for a in alphas:
                if not view.closed(a.timeframe):
                    continue
                variants = ({}, *PARAM_VARIANTS.get(a.name, ()), *AGNOSTIC_VARIANTS.get(a.name, ()))
                for v, over in enumerate(variants):
                    s = a.evaluate(view, ctx, {**params.alphas[a.name], **over})
                    if s is None:
                        continue
                    assert s.symbol == symbol, (a.name, s.symbol)
                    prices = tuple(None if x is None else x / scale for x in
                                   (s.stop, s.take_profit, s.tp1, s.limit_price, s.atr, s.tags.get("ref_price")))
                    out[a.name].append((t, up, v, s.side, s.entry_style, s.timeframe, s.max_hold_bars, s.confidence)
                                       + prices)
    return out


def _same(a: tuple, b: tuple) -> bool:
    if a[:7] != b[:7]:
        return False
    for x, y in zip(a[7:], b[7:]):
        if (x is None) != (y is None):
            return False
        if x is not None and not math.isclose(x, y, rel_tol=REL, abs_tol=0.0):
            return False
    return True


def agnostic_findings(base: dict, other: dict) -> dict[str, str]:
    """alpha -> first difference between two decision records (empty when every alpha agrees)."""
    out = {}
    for name in sorted(set(base) | set(other)):
        a, b = base.get(name, []), other.get(name, [])
        if len(a) != len(b) or not all(_same(x, y) for x, y in zip(a, b)):
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if not _same(x, y)), min(len(a), len(b)))
            out[name] = (f"{len(a)} vs {len(b)} decisions; first difference at #{first}: "
                         f"{a[first] if first < len(a) else None} vs {b[first] if first < len(b) else None}")
    return out


@pytest.fixture(scope="module")
def agnostic_runs():
    """The 1h-close drive over LONG_DAYS (every alpha, every timeframe close) and the 15m drive over DAYS (the
    intraday alphas at every quarter hour), each as the original, renamed and price-scaled market."""
    runs = {}
    long_ca = long_candles()
    short_ca = synth_candles(DAYS * 1440, seed=5, price=100.0, vol=0.0022)
    for drive, ca, every in (("1h", long_ca, 60), ("15m", short_ca, 15)):
        runs[(drive, "base")] = decisions(ca, SYM, 1.0, every)
        runs[(drive, "renamed")] = decisions(ca, RENAMED, 1.0, every)
        for sc in AGNOSTIC_SCALES:
            runs[(drive, sc)] = decisions(scaled_candles(ca, sc), SYM, sc, every)
    return runs


def test_agnostic_drives_exercise_the_alphas(agnostic_runs):
    fired = {a for drive in ("1h", "15m") for a, v in agnostic_runs[(drive, "base")].items() if v}
    assert len(fired) >= 10, sorted(fired)


def test_decisions_do_not_depend_on_the_symbol_name(agnostic_runs):
    for drive in ("1h", "15m"):
        assert agnostic_findings(agnostic_runs[(drive, "base")], agnostic_runs[(drive, "renamed")]) == {}, drive


@pytest.mark.parametrize("scale", AGNOSTIC_SCALES)
def test_decisions_do_not_depend_on_the_price_level(agnostic_runs, scale):
    for drive in ("1h", "15m"):
        assert agnostic_findings(agnostic_runs[(drive, "base")], agnostic_runs[(drive, scale)]) == {}, (drive, scale)


def test_every_alpha_is_checked_somewhere(agnostic_runs):
    """Every registered alpha decides at least once in the drives (shipped, researched or permissive settings), so no
    invariance above holds vacuously."""
    fired = {a for k, run in agnostic_runs.items() if k[1] == "base" for a, v in run.items() if v}
    assert fired == {a.name for a in ALL_ALPHAS}, sorted({a.name for a in ALL_ALPHAS} - fired)


@pytest.mark.parametrize("scale", (1000.0, 0.001))
def test_discovery_features_are_price_level_free(scale):
    """The mined-rule feature vector (FEATURES_X, the discovered alpha's and the miner's inputs) does not move with the
    price level: a rule threshold learned on one coin means the same on any other."""
    from heartless.learning.discovery import FEATURES_X, decision_features

    ca = long_candles(40)

    def feats(c: CandleArrays) -> tuple[np.ndarray, np.ndarray]:
        view = MarketView(SYM)
        view.rebuild(c, live=False)
        ct = c.view("close_time")
        rows, ts = [], []
        for j in range(59, c.n, 60):
            t = int(ct[j])
            view.seek(t)
            ex, _, _ = moving_extras(t, True)
            rows.append(decision_features(view, ex, market(t, True), t))
            ts.append(t)
        return np.array(rows), np.array(ts)

    (a, ts), (b, _) = feats(ca), feats(scaled_candles(ca, scale))
    first_bar = (ts + 1) % MS_DAY == MS_HOUR  # the 1h close that ends the first hour of a UTC session
    bad = {}
    for i, name in enumerate(FEATURES_X):
        x, y = a[:, i], b[:, i]
        same = np.isclose(x, y, rtol=FEATURE_RTOL, atol=FEATURE_ATOL) | (np.isnan(x) & np.isnan(y))
        if name in NUMERIC_FEATURES:
            same |= first_bar
        if not same.all():
            k = int(np.flatnonzero(~same)[0])
            bad[name] = (int((~same).sum()), float(x[k]), float(y[k]))
    assert bad == {}, bad

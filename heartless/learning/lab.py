"""Research lab: fast, parallel, honest evaluation of alphas on stored (real) history.

* Each symbol is backtested independently in its own worker process with the production engine, paper
  simulator and alpha code (no research-only re-implementation that could drift from live behaviour).
* Results are aggregated per alpha, per symbol and per calendar month, with R-multiple statistics that are
  comparable across symbols and account sizes.
* `splits()` divides the stored history into TRAIN / VALID / HOLDOUT. Exploration should only look at TRAIN,
  model selection at VALID, and HOLDOUT is touched once, at the very end, to estimate live performance.

CLI: `heartless lab --alpha trend_pullback --split train --set trend_pullback.adx_min=24`
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from heartless.core.models import SymbolInfo
from heartless.execution.stats import summarize
from heartless.util.timeutil import MS_DAY

WARMUP_MS = 12 * MS_DAY  # indicator warm-up loaded before the evaluation window (1h EMA200 needs ~8.3 days)
# Alpha research measures the raw edge of the signals, so the account-level circuit breakers (daily/weekly loss
# halts, drawdown pause) are switched off: otherwise a losing stretch silences an alpha for the rest of the
# window and its statistics stop describing the signal. Per-trade sizing and per-symbol limits stay as in live.
RESEARCH_SETTINGS = {"DAILY_LOSS_LIMIT_PCT": 1000.0, "WEEKLY_LOSS_LIMIT_PCT": 1000.0, "MAX_DRAWDOWN_HALT_PCT": 1000.0}
# cost stress test: 1.5x fees and 2x slippage; an edge that only exists at nominal costs is not an edge
STRESS_SETTINGS = {"TAKER_FEE": 0.00075, "MAKER_FEE": 0.0003, "BACKTEST_SLIPPAGE_BPS": 3.0}


# --- symbol metadata -----------------------------------------------------------------------------------
def infer_symbol_info(symbol: str, close: np.ndarray, volume: np.ndarray | None = None) -> SymbolInfo:
    """Approximate exchange filters from the data itself (used when exchangeInfo is unavailable offline)."""
    px = np.unique(np.round(close[-50_000:], 8))
    d = np.diff(px)
    d = d[d > 0]
    tick = float(np.round(np.min(d), 8)) if len(d) else 0.01
    # normalise to a power-of-ten-ish tick (0.1, 0.01, 0.0001, ...)
    if tick > 0:
        mag = 10 ** math.floor(math.log10(tick))
        tick = round(round(tick / mag) * mag, 10)
    price = float(np.median(close[-10_000:])) if len(close) else 1.0
    # quantity step worth roughly $1-$10 (BTC ~0.0001, ETH ~0.001, DOGE ~10): fine enough for any account size
    step = 10.0 ** math.floor(math.log10(10.0 / max(price, 1e-9)))
    step = float(f"{min(max(step, 1e-6), 1000.0):.10f}")
    return SymbolInfo(symbol, symbol[:-4], "USDT", tick_size=tick, step_size=step, min_qty=step, min_notional=5.0,
                      price_precision=max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0,
                      quantity_precision=max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0)


# --- splits -------------------------------------------------------------------------------------------
def _ms(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def splits(first_ms: int, last_ms: int, holdout_days: int = 35, valid_days: int = 62) -> dict[str, tuple[int, int]]:
    """TRAIN = oldest history after warm-up, VALID = the next `valid_days`, HOLDOUT = the newest `holdout_days`."""
    hold_start = last_ms - holdout_days * MS_DAY
    valid_start = hold_start - valid_days * MS_DAY
    train_start = first_ms + WARMUP_MS
    return {"train": (train_start, valid_start - 1), "valid": (valid_start, hold_start - 1),
            "holdout": (hold_start, last_ms), "all": (train_start, last_ms)}


def store_splits(store, symbols: list[str]) -> dict[str, tuple[int, int]]:
    ranges = [store.candle_range(s) for s in symbols]
    firsts = [r[0] for r in ranges if r[0] is not None]
    lasts = [r[1] for r in ranges if r[1] is not None]
    if not firsts:
        raise RuntimeError("no candles stored for the requested symbols; run `heartless fetch` first")
    return splits(max(firsts), min(lasts))


# --- worker ------------------------------------------------------------------------------------------
def _run_symbol(job: dict) -> dict:
    """Backtest one symbol in a worker process. Must stay a top-level function (picklable)."""
    import logging

    logging.disable(logging.WARNING)
    from heartless.config import Settings
    from heartless.core.store import Store
    from heartless.data.candles import CandleArrays
    from heartless.data.extras import MetricsSeries
    from heartless.learning.backtester import FUNDING_INTERVAL, Backtester
    from heartless.strategy.params import StrategyParams

    t0 = time.time()
    store = Store(job["db_path"])
    sym = job["symbol"]
    start, end = job["start"], job["end"]
    rows = store.load_candles(sym, start=start - job.get("warmup_ms", WARMUP_MS), end=end)
    if len(rows) < 2000:
        store.close()
        return {"symbol": sym, "error": "not enough candles", "trades": [], "stats": summarize([])}
    ca = CandleArrays("1m", capacity=len(rows) + 16)
    ca.extend(rows)
    info = SymbolInfo(**job["info"]) if job.get("info") else infer_symbol_info(sym, ca.view("close"))
    funding = {sym: store.load_funding(sym, start - job.get("warmup_ms", WARMUP_MS) - FUNDING_INTERVAL, end + FUNDING_INTERVAL)}
    mrows = store.load_metrics(sym, start - job.get("warmup_ms", WARMUP_MS), end) if hasattr(store, "load_metrics") else []
    metrics = {sym: MetricsSeries.from_rows(mrows)} if mrows else {}
    store.close()
    settings = Settings(_env_file=None, **{**RESEARCH_SETTINGS, **job.get("settings", {})})
    bt = Backtester(settings, {sym: info}, {sym: ca}, funding, metrics)
    params = StrategyParams.from_dict(job["params"])
    res = bt.run(params, start, end, only_alpha=job.get("only_alpha"), initial_balance=job.get("balance", 10_000.0))
    keep = ("symbol", "side", "alpha", "alphas", "regime", "entry_time", "exit_time", "entry_price", "exit_price", "pnl",
            "gross", "fees", "funding", "r_multiple", "risk_amount", "exit_reason", "bars_held", "max_fav_r", "max_adv_r",
            "confidence", "notional")
    trades = [{k: t.get(k) for k in keep} for t in res.trades]
    return {"symbol": sym, "stats": res.stats, "trades": trades, "skipped": res.skipped, "seconds": time.time() - t0,
            "bars": int(ca.n), "info": asdict(info)}


# --- aggregation -------------------------------------------------------------------------------------
@dataclass
class LabResult:
    params_version: str
    alpha: str | None
    start: int
    end: int
    overall: dict
    by_alpha: dict
    by_symbol: dict
    by_month: dict
    trades: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    errors: dict = field(default_factory=dict)

    def consistency(self) -> dict:
        months = [m for m in self.by_month.values() if m["n"] >= 3]
        syms = [s for s in self.by_symbol.values() if s["n"] >= 3]
        return {"months_positive": sum(1 for m in months if m["avg_r"] > 0), "months": len(months),
                "symbols_positive": sum(1 for s in syms if s["avg_r"] > 0), "symbols": len(syms)}

    def summary(self) -> dict:
        o = self.overall
        return {"alpha": self.alpha, "n": o["n"], "win_rate": round(o["win_rate"], 1),
                "avg_r": round(o["avg_r"], 4), "t_stat": round(o["t_stat"], 2),
                "profit_factor": round(min(o["profit_factor"], 99.0), 3), "net": round(o["net"], 1),
                "fees": round(o["fees"], 1), "trades_per_day": round(o["n"] / max((self.end - self.start) / MS_DAY, 1e-9), 2),
                "avg_hold_min": round(o["avg_hold_min"], 1), **self.consistency(), "seconds": round(self.seconds, 1)}

    def report(self) -> str:
        s = self.summary()
        lines = [f"[{self.alpha or 'ensemble'}] {datetime.fromtimestamp(self.start / 1000, tz=timezone.utc):%Y-%m-%d} -> "
                 f"{datetime.fromtimestamp(self.end / 1000, tz=timezone.utc):%Y-%m-%d}  n={s['n']} win={s['win_rate']}% "
                 f"avgR={s['avg_r']:+.4f} t={s['t_stat']:+.2f} PF={s['profit_factor']:.2f} net={s['net']:+.1f} "
                 f"fees={s['fees']:.0f} trades/day={s['trades_per_day']} hold={s['avg_hold_min']:.0f}m "
                 f"months+ {s['months_positive']}/{s['months']} symbols+ {s['symbols_positive']}/{s['symbols']} ({s['seconds']:.0f}s)"]
        for a, st in sorted(self.by_alpha.items(), key=lambda kv: -kv[1]["avg_r"]):
            lines.append(f"  alpha {a:20s} n={st['n']:5d} win={st['win_rate']:5.1f}% avgR={st['avg_r']:+.4f} "
                         f"t={st['t_stat']:+.2f} PF={min(st['profit_factor'], 99):5.2f}")
        for sym, st in sorted(self.by_symbol.items()):
            lines.append(f"  sym   {sym:12s} n={st['n']:5d} avgR={st['avg_r']:+.4f} PF={min(st['profit_factor'], 99):5.2f} net={st['net']:+.1f}")
        for m, st in sorted(self.by_month.items()):
            lines.append(f"  month {m} n={st['n']:5d} avgR={st['avg_r']:+.4f} PF={min(st['profit_factor'], 99):5.2f}")
        reasons = defaultdict(list)
        for t in self.trades:
            reasons[t.get("exit_reason") or "?"].append(t.get("r_multiple") or 0.0)
        for r, v in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"  exit  {r:24s} n={len(v):5d} avgR={float(np.mean(v)):+.3f}")
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        return "\n".join(lines)


def _group(trades: list[dict], key) -> dict:
    g: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        g[key(t)].append(t)
    return {k: summarize(v) for k, v in g.items()}


def evaluate(params, symbols: list[str], start: int, end: int, db_path: str, only_alpha: str | None = None,
             workers: int | None = None, settings: dict | None = None, infos: dict | None = None,
             pool: ProcessPoolExecutor | None = None) -> LabResult:
    """Backtest `params` on every symbol over [start, end] in parallel and aggregate the results."""
    t0 = time.time()
    pdict = params.to_dict() if hasattr(params, "to_dict") else dict(params)
    jobs = [{"db_path": str(db_path), "symbol": s, "start": int(start), "end": int(end), "params": pdict,
             "only_alpha": only_alpha, "settings": settings or {}, "info": (infos or {}).get(s)} for s in symbols]
    own = pool is None
    if own:
        n = workers or max(1, min(len(jobs), (os.cpu_count() or 2)))
        pool = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
    try:
        results = list(pool.map(_run_symbol, jobs))
    finally:
        if own:
            pool.shutdown()
    trades = [t for r in results for t in r["trades"]]
    errors = {r["symbol"]: r["error"] for r in results if r.get("error")}

    def month(t: dict) -> str:
        return datetime.fromtimestamp((t.get("exit_time") or 0) / 1000, tz=timezone.utc).strftime("%Y-%m")

    return LabResult(params_version=pdict.get("version", "?"), alpha=only_alpha, start=int(start), end=int(end),
                     overall=summarize(trades), by_alpha=_group(trades, lambda t: t.get("alpha") or "?"),
                     by_symbol=_group(trades, lambda t: t.get("symbol") or "?"), by_month=_group(trades, month),
                     trades=trades, seconds=time.time() - t0, errors=errors)


def apply_overrides(params, overrides: list[str]):
    """`alpha.param=value` / `ensemble.key=value` / `enabled.alpha=0|1` overrides on a StrategyParams copy."""
    p = params.clone(version=params.version + "+lab", note="lab overrides")
    for item in overrides or []:
        key, _, raw = item.partition("=")
        scope, _, name = key.partition(".")
        val: float | bool = float(raw)
        if scope == "ensemble":
            p.ensemble[name] = val
        elif scope == "enabled":
            p.enabled[name] = bool(int(val))
        elif scope in p.alphas:
            p.alphas[scope][name] = val
        else:
            raise KeyError(f"unknown override scope {scope!r}")
    return p


def main_cli(settings, args) -> None:
    from heartless.core.store import Store
    from heartless.strategy.params import StrategyParams

    store = Store(settings.db_path)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else store.candle_symbols()
    sp = store_splits(store, symbols)
    if args.start or args.end:
        start = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.start else sp["all"][0]
        end = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp() * 1000) if args.end else sp["all"][1]
    else:
        start, end = sp[args.split]
    if args.params:
        with open(args.params) as f:
            params = StrategyParams.from_dict(json.load(f))
    else:
        rows = store.load_params_versions(role="champion", limit=1)
        params = StrategyParams.from_dict(rows[0]["params"]) if rows else StrategyParams.default()
    store.close()
    params = apply_overrides(params, args.set)
    extra_settings = dict(STRESS_SETTINGS) if getattr(args, "stress", False) else {}
    res = evaluate(params, symbols, start, end, str(settings.db_path), only_alpha=args.alpha, workers=args.workers,
                   settings=extra_settings)
    if args.json:
        print(json.dumps({"summary": res.summary(), "by_alpha": res.by_alpha, "by_symbol": res.by_symbol,
                          "by_month": res.by_month}, default=str))
    else:
        print(res.report())
    if args.trades_out:
        with open(args.trades_out, "w") as f:
            json.dump(res.trades, f, default=str)

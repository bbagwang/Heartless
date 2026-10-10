"""Research lab: fast, parallel, honest evaluation of alphas on stored (real) history.

* Each symbol is backtested independently in its own worker process with the production engine, paper
  simulator and alpha code (no research-only re-implementation that could drift from live behaviour).
* Results are aggregated per alpha, per symbol and per calendar month, with R-multiple statistics that are
  comparable across symbols and account sizes.
* `splits()` divides the stored history into TRAIN / VALID / HOLDOUT. Exploration should only look at TRAIN,
  model selection at VALID, and HOLDOUT is touched once, at the very end, to estimate live performance.
* Long windows (years) are cut into (symbol x chunk) jobs, calendar years by default, each with its own warm-up, so
  memory stays bounded and every core stays busy. Chunks hand off through a burn-in / run-off overlap and own the
  trades entered inside them (nothing is counted twice); a trade still open when an interior chunk's run-off ends is
  force-closed and flagged `boundary`, and reports show the statistics with and without those trades.
* Every evaluated window passes the seal check (`heartless.learning.seal`): sealed periods are refused unless
  explicitly unsealed.

CLI: `heartless lab --alpha trend_pullback --split train --set trend_pullback.adx_min=24`
     `heartless lab --alpha htf_trend --years 2022,2023,2024 --by-side --out docs/results/htf_2022_2024.json`
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
from heartless.learning import seal
from heartless.util.timeutil import MS_DAY, MS_MINUTE

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


# --- windows ------------------------------------------------------------------------------------------
def _ms(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def parse_when(text: str, end: bool = False) -> int:
    """ISO date or date-time (UTC unless an offset is given) to epoch ms.

    A date-only END is exclusive: `--end 2025-01-01` evaluates through 2024-12-31 23:59:59.999 (the same bars as
    the old inclusive midnight, whose minute closes after it), so a window that stops at a sealed period's start
    does not touch it."""
    raw = text.strip()
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    return ms - 1 if end and len(raw) == 10 else ms


def year_window(years: str) -> tuple[int, int, list[int]]:
    """`2022,2023,2024` -> (2022-01-01 00:00, 2024-12-31 23:59:59.999, [2022, 2023, 2024]); contiguous years only."""
    try:
        ys = sorted({int(y) for y in years.split(",") if y.strip()})
    except ValueError:
        raise ValueError(f"--years expects comma separated years, got {years!r}") from None
    if not ys:
        raise ValueError("--years is empty")
    if ys != list(range(ys[0], ys[-1] + 1)):
        raise ValueError(f"--years must be contiguous (the evaluated window is their union), got {ys}")
    return _ms(ys[0], 1, 1), _ms(ys[-1] + 1, 1, 1) - 1, ys


def chunk_windows(start: int, end: int, chunk_days: int | None = None) -> list[tuple[int, int]]:
    """Disjoint evaluation windows covering [start, end]: UTC calendar years by default, consecutive `chunk_days`
    blocks from `start` if given, or the whole window for 0."""
    if chunk_days is not None and chunk_days <= 0:
        return [(start, end)]
    bounds: list[int] = []
    if chunk_days is None:
        y = datetime.fromtimestamp(start / 1000, tz=timezone.utc).year + 1
        while _ms(y, 1, 1) < end:
            bounds.append(_ms(y, 1, 1))
            y += 1
    else:
        b = start + chunk_days * MS_DAY
        while b < end:
            bounds.append(b)
            b += chunk_days * MS_DAY
    edges = [start] + bounds
    return [(a, edges[i + 1] - 1 if i + 1 < len(edges) else end) for i, a in enumerate(edges)]


# --- splits -------------------------------------------------------------------------------------------


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
BOUNDARY_REASON = "backtest end"  # exit reason of the forced flatten at the end of a backtest run
# Chunk hand-off. A chunk keeps the trades ENTERED inside its window, but simulates a little more on both sides so the
# cut is (almost) invisible: it runs on after its end with exits only mattering (positions opened in the window close
# naturally instead of being force-closed; later entries belong to the next chunk and are dropped), and every chunk
# but the first starts BURN_IN_MS early so a position the continuous run would hold across the cut is re-created and
# blocks the same entries (its trade belongs to the previous chunk and is dropped here). Both exceed every alpha's
# time stop (htf_trend's maximum is 96h).
BURN_IN_MS = 7 * MS_DAY
RUNOFF_MS = 7 * MS_DAY


def _run_symbol(job: dict) -> dict:
    """Backtest one symbol over one evaluation window in a worker process. Must stay a top-level function (picklable).

    The window [start, end] may be one chunk of a longer evaluation (see BURN_IN_MS / RUNOFF_MS): only trades entered
    inside it are returned, so a position is counted exactly once, by the chunk it was opened in. A returned trade that
    was still open when an interior chunk's run-off ended is force-closed there and flagged `boundary`. A symbol whose
    stored history starts inside the warm-up (listed later) is evaluated only after a full warm-up from its first
    candle."""
    import logging

    logging.disable(logging.WARNING)
    from heartless.config import Settings
    from heartless.core.store import Store
    from heartless.data.candles import CandleArrays
    from heartless.data.extras import MetricsSeries
    from heartless.learning.backtester import FUNDING_INTERVAL, Backtester
    from heartless.strategy.params import StrategyParams

    t0 = time.time()
    sym = job["symbol"]
    start, end = job["start"], job["end"]
    unseal = job.get("unseal", ())
    seal.check_window(start, end, unseal)  # also here: a worker job must never be a way around the seal
    warm = job.get("warmup_ms", WARMUP_MS)
    label = job.get("label", "")
    first_chunk, final = job.get("first", True), job.get("final", True)
    sim_start = start - (0 if first_chunk else job.get("burn_in_ms", 0))
    # the run-off past an interior chunk's end never reads a sealed period
    sim_end = end if final else seal.readable_until(end, end + job.get("runoff_ms", 0), unseal)
    base = {"symbol": sym, "chunk": label, "start": start, "end": end, "trades": [], "stats": summarize([])}
    store = Store(job["db_path"])
    rows = store.load_candles(sym, start=sim_start - warm, end=sim_end)
    if rows and rows[0].open_time > sim_start - warm + MS_DAY:
        sim_start = max(sim_start, rows[0].open_time + warm)  # history begins inside the warm-up: wait for a full one
    eff_start = max(start, sim_start)
    if len(rows) < 2000 or eff_start >= end:
        store.close()
        return {**base, "error": "not enough candles"}
    ca = CandleArrays("1m", capacity=len(rows) + 16)
    ca.extend(rows)
    del rows
    info = SymbolInfo(**job["info"]) if job.get("info") else infer_symbol_info(sym, ca.view("close"))
    # the next settlement's rate is read up to one interval past the simulated end, but never inside a sealed period
    fund_end = seal.readable_until(sim_end, sim_end + FUNDING_INTERVAL, unseal)
    funding = {sym: store.load_funding(sym, sim_start - warm - FUNDING_INTERVAL, fund_end)}
    mrows = store.load_metrics(sym, sim_start - warm, sim_end) if hasattr(store, "load_metrics") else []
    metrics = {sym: MetricsSeries.from_rows(mrows)} if mrows else {}
    store.close()
    settings = Settings(_env_file=None, **{**RESEARCH_SETTINGS, **job.get("settings", {})})
    bt = Backtester(settings, {sym: info}, {sym: ca}, funding, metrics)
    params = StrategyParams.from_dict(job["params"])
    res = bt.run(params, sim_start, sim_end, only_alpha=job.get("only_alpha"), initial_balance=job.get("balance", 10_000.0))
    keep = ("symbol", "side", "alpha", "alphas", "regime", "entry_time", "exit_time", "entry_price", "exit_price", "pnl",
            "gross", "fees", "funding", "r_multiple", "risk_amount", "exit_reason", "bars_held", "max_fav_r", "max_adv_r",
            "confidence", "notional")
    trades = []
    for t in res.trades:
        et = int(t.get("entry_time") or 0)
        if (not first_chunk and et < start) or (not final and et > end):
            continue  # burn-in replica (owned by the previous chunk) or a run-off entry (owned by the next one)
        d = {k: t.get(k) for k in keep}
        d["boundary"] = bool(not final and t.get("exit_reason") == BOUNDARY_REASON)
        if label:
            d["chunk"] = label
        trades.append(d)
    return {**base, "eff_start": eff_start, "sim": [sim_start, sim_end], "stats": summarize(trades), "trades": trades,
            "skipped": res.skipped, "seconds": time.time() - t0, "bars": int(ca.n), "info": asdict(info)}


# --- aggregation -------------------------------------------------------------------------------------
def day_clustered_t(trades: list[dict]) -> float:
    """t-statistic with trades clustered by exit day: simultaneous trades on correlated coins are not independent,
    so this is the honest significance (the per-trade t_stat overstates it)."""
    if len(trades) < 3:
        return 0.0
    r = np.array([float(t.get("r_multiple") or 0.0) for t in trades])
    days = np.array([int((t.get("exit_time") or 0) // MS_DAY) for t in trades])
    uniq, inv = np.unique(days, return_inverse=True)
    k = len(uniq)
    if k < 5:
        return 0.0
    mean = r.mean()
    resid = np.bincount(inv, weights=r) - np.bincount(inv) * mean
    var = (resid ** 2).sum() * k / (k - 1) / (len(r) ** 2)
    return float(mean / math.sqrt(var)) if var > 0 else 0.0


def _month(t: dict) -> str:
    return datetime.fromtimestamp((t.get("exit_time") or 0) / 1000, tz=timezone.utc).strftime("%Y-%m")


def _year(t: dict) -> str:
    """Trades belong to the year they were opened in (the window that made the decision)."""
    return str(datetime.fromtimestamp((t.get("entry_time") or 0) / 1000, tz=timezone.utc).year)


def months_positive(trades: list[dict], min_n: int = 3) -> tuple[int, int]:
    """(months with avg R > 0, months with at least `min_n` trades), months by exit time."""
    g: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        g[_month(t)].append(float(t.get("r_multiple") or 0.0))
    months = [v for v in g.values() if len(v) >= min_n]
    return sum(1 for v in months if sum(v) > 0), len(months)


def group_stats(trades: list[dict]) -> dict:
    """Compact per-group statistics used for the year / side breakdowns."""
    o = summarize(trades)
    mp, m = months_positive(trades)
    return {"n": o["n"], "win_rate": round(o["win_rate"], 1), "avg_r": round(o["avg_r"], 4),
            "profit_factor": round(min(o["profit_factor"], 99.0), 3), "t_day": round(day_clustered_t(trades), 2),
            "net": round(o["net"], 1), "months_positive": mp, "months": m}


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
    chunks: list = field(default_factory=list)  # [(start, end)] evaluation windows
    jobs: list = field(default_factory=list)  # per (symbol, chunk): timing, bars, trades, effective start

    def consistency(self) -> dict:
        months = [m for m in self.by_month.values() if m["n"] >= 3]
        syms = [s for s in self.by_symbol.values() if s["n"] >= 3]
        return {"months_positive": sum(1 for m in months if m["avg_r"] > 0), "months": len(months),
                "symbols_positive": sum(1 for s in syms if s["avg_r"] > 0), "symbols": len(syms)}

    def day_clustered_t(self) -> float:
        return day_clustered_t(self.trades)

    @property
    def boundary_trades(self) -> list[dict]:
        return [t for t in self.trades if t.get("boundary")]

    def by_year(self) -> dict:
        return {k: group_stats(v) for k, v in sorted(_bucket(self.trades, _year).items())}

    def by_side(self) -> dict:
        return {k: group_stats(v) for k, v in sorted(_bucket(self.trades, lambda t: t.get("side") or "?").items())}

    def by_year_side(self) -> dict:
        return {k: group_stats(v) for k, v in
                sorted(_bucket(self.trades, lambda t: f"{_year(t)} {t.get('side') or '?'}").items())}

    def summary(self) -> dict:
        o = self.overall
        return {"alpha": self.alpha, "n": o["n"], "win_rate": round(o["win_rate"], 1),
                "avg_r": round(o["avg_r"], 4), "t_stat": round(o["t_stat"], 2), "t_day": round(self.day_clustered_t(), 2),
                "profit_factor": round(min(o["profit_factor"], 99.0), 3), "net": round(o["net"], 1),
                "fees": round(o["fees"], 1), "trades_per_day": round(o["n"] / max((self.end - self.start) / MS_DAY, 1e-9), 2),
                "avg_hold_min": round(o["avg_hold_min"], 1), **self.consistency(), "seconds": round(self.seconds, 1),
                "boundary": len(self.boundary_trades), "chunks": len(self.chunks) or 1}

    def summary_excl_boundary(self) -> dict:
        return group_stats([t for t in self.trades if not t.get("boundary")])

    def to_json(self) -> dict:
        """Everything a result file needs (the CLI adds run metadata: overrides, commit, seal status, argv)."""
        return {"window": {"start": _iso(self.start), "end": _iso(self.end), "start_ms": self.start, "end_ms": self.end},
                "summary": self.summary(), "summary_excl_boundary": self.summary_excl_boundary(),
                "by_year": self.by_year(), "by_alpha": self.by_alpha, "by_symbol": self.by_symbol,
                "by_month": self.by_month, "by_side": self.by_side(), "by_year_side": self.by_year_side(),
                "chunks": [[_iso(a), _iso(b)] for a, b in self.chunks], "jobs": self.jobs, "errors": self.errors}

    def report(self, by_side: bool = False) -> str:
        s = self.summary()
        lines = [f"[{self.alpha or 'ensemble'}] {_iso(self.start)[:10]} -> {_iso(self.end)[:10]}  n={s['n']} "
                 f"win={s['win_rate']}% avgR={s['avg_r']:+.4f} t={s['t_stat']:+.2f} PF={s['profit_factor']:.2f} "
                 f"net={s['net']:+.1f} t_day={s['t_day']:+.2f} fees={s['fees']:.0f} trades/day={s['trades_per_day']} "
                 f"hold={s['avg_hold_min']:.0f}m months+ {s['months_positive']}/{s['months']} symbols+ "
                 f"{s['symbols_positive']}/{s['symbols']} ({s['seconds']:.0f}s)"]
        if len(self.chunks) > 1:
            x = self.summary_excl_boundary()
            lines.append(f"  chunks {len(self.chunks)} x {len(self.by_symbol) or '-'} symbols, boundary trades "
                         f"{s['boundary']}; excl. boundary n={x['n']} avgR={x['avg_r']:+.4f} PF={x['profit_factor']:.2f} "
                         f"t_day={x['t_day']:+.2f}")

        def line(tag: str, key: str, st: dict) -> str:
            return (f"  {tag:5s} {key:12s} n={st['n']:5d} avgR={st['avg_r']:+.4f} PF={st['profit_factor']:5.2f} "
                    f"t_day={st['t_day']:+.2f} months+ {st['months_positive']}/{st['months']}")

        for y, st in self.by_year().items():
            lines.append(line("year", y, st))
        if by_side:
            for k, st in self.by_side().items():
                lines.append(line("side", k, st))
            for k, st in self.by_year_side().items():
                lines.append(line("ys", k, st))
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


def _bucket(trades: list[dict], key) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        g[key(t)].append(t)
    return g


def _group(trades: list[dict], key) -> dict:
    return {k: summarize(v) for k, v in _bucket(trades, key).items()}


def evaluate(params, symbols: list[str], start: int, end: int, db_path: str, only_alpha: str | None = None,
             workers: int | None = None, settings: dict | None = None, infos: dict | None = None,
             pool: ProcessPoolExecutor | None = None, chunk_days: int | None = None, unseal=(),
             warmup_ms: int = WARMUP_MS, burn_in_ms: int = BURN_IN_MS, runoff_ms: int = RUNOFF_MS) -> LabResult:
    """Backtest `params` on every symbol over [start, end] in parallel and aggregate the results.

    The window is cut into chunks (calendar years by default, `chunk_days` blocks, or none for 0); each
    (symbol, chunk) is an independent job with its own warm-up and the burn-in / run-off hand-off described at
    BURN_IN_MS. Raises SealedError when [start, end] overlaps a sealed period that is not in `unseal`."""
    seal.check_window(int(start), int(end), unseal)
    unseal = seal.opened(int(start), int(end), unseal)  # an unseal the window does not need opens nothing
    t0 = time.time()
    pdict = params.to_dict() if hasattr(params, "to_dict") else dict(params)
    windows = chunk_windows(int(start), int(end), chunk_days)
    multi = len(windows) > 1

    def label(a: int) -> str:
        if not multi:
            return ""
        return str(datetime.fromtimestamp(a / 1000, tz=timezone.utc).year) if chunk_days is None else _iso(a)[:10]

    jobs = [{"db_path": str(db_path), "symbol": s, "start": a, "end": b, "params": pdict, "only_alpha": only_alpha,
             "settings": settings or {}, "info": (infos or {}).get(s), "label": label(a), "first": k == 0,
             "final": k == len(windows) - 1, "unseal": sorted(unseal or ()), "warmup_ms": warmup_ms,
             "burn_in_ms": burn_in_ms, "runoff_ms": runoff_ms}
            for s in symbols for k, (a, b) in enumerate(windows)]
    own = pool is None
    if own:
        n = workers or max(1, min(len(jobs), (os.cpu_count() or 2)))
        pool = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
    try:
        results = list(pool.map(_run_symbol, jobs))
    finally:
        if own:
            pool.shutdown()
    trades: list[dict] = []
    stray = 0
    for job, r in zip(jobs, results):
        for t in r["trades"]:
            # every trade belongs to exactly one window: the one its entry falls in (chunks are disjoint)
            if job["start"] - MS_MINUTE <= int(t.get("entry_time") or 0) <= job["end"]:
                trades.append(t)
            else:
                stray += 1
    errors = {(f"{r['symbol']}@{r['chunk']}" if r.get("chunk") else r["symbol"]): r["error"]
              for r in results if r.get("error")}
    if stray:
        errors["stray_trades"] = f"{stray} trade(s) outside their chunk window dropped"
    job_info = [{"symbol": r["symbol"], "chunk": r.get("chunk", ""), "start": _iso(r["start"]),
                 "eff_start": _iso(r.get("eff_start", r["start"])), "end": _iso(r["end"]),
                 "simulated": [_iso(x) for x in r["sim"]] if r.get("sim") else None, "n": len(r["trades"]),
                 "bars": r.get("bars", 0), "seconds": round(r.get("seconds", 0.0), 1), "error": r.get("error")}
                for r in results]
    return LabResult(params_version=pdict.get("version", "?"), alpha=only_alpha, start=int(start), end=int(end),
                     overall=summarize(trades), by_alpha=_group(trades, lambda t: t.get("alpha") or "?"),
                     by_symbol=_group(trades, lambda t: t.get("symbol") or "?"), by_month=_group(trades, _month),
                     trades=trades, seconds=time.time() - t0, errors=errors, chunks=windows, jobs=job_info)


def apply_overrides(params, overrides: list[str], notes: list | None = None):
    """`alpha.param=value` / `ensemble.key=value` / `enabled.alpha=0|1` overrides on a StrategyParams copy.

    Alpha parameters only run on their ParamSpec grid (StrategyParams.from_dict -> ParamSpec.clip in the worker
    rounds to `step`, clips to [lo, hi], snaps to `choices`, and keeps tp1_r < tp_r). A requested value that would
    silently run as a different one is stored as the effective value and reported on stderr (and in `notes`)."""
    import sys

    from heartless.strategy.params import ALPHA_SPECS, _enforce_order

    p = params.clone(version=params.version + "+lab", note="lab overrides")
    touched: set[str] = set()

    def warn(key: str, requested, effective, why: str) -> None:
        print(f"warning: --set {key}={requested} {why}; it runs as {key}={effective}", file=sys.stderr)
        if notes is not None:
            notes.append({"key": key, "requested": requested, "effective": effective, "why": why})

    for item in overrides or []:
        key, _, raw = item.partition("=")
        scope, _, name = key.partition(".")
        val: float | bool = float(raw)
        if scope == "ensemble":
            p.ensemble[name] = val
        elif scope == "enabled":
            p.enabled[name] = bool(int(val))
        elif scope in p.alphas:
            spec = next((s for s in ALPHA_SPECS.get(scope, ()) if s.name == name), None)
            if spec is None:
                print(f"warning: --set {key}: {scope} has no parameter {name!r}; the alpha ignores it", file=sys.stderr)
                if notes is not None:
                    notes.append({"key": key, "requested": val, "effective": None, "why": "unknown parameter"})
            else:
                eff = spec.clip(val)
                if eff != val:
                    why = (f"is outside [{spec.lo:g}, {spec.hi:g}]" if not spec.choices and not spec.lo <= val <= spec.hi
                           else f"is not on the grid ({'choices ' + str(spec.choices) if spec.choices else 'step ' + format(spec.step, 'g')})")
                    warn(key, raw, eff, why)
                val = eff
                touched.add(scope)
            p.alphas[scope][name] = val
        else:
            raise KeyError(f"unknown override scope {scope!r}")
    for scope in touched:
        before = dict(p.alphas[scope])
        _enforce_order(scope, p.alphas[scope])
        for k, v in p.alphas[scope].items():
            if before[k] != v:
                warn(f"{scope}.{k}", before[k], v, "must stay below tp_r")
    return p


def _clean(obj):
    """JSON-safe copy: non-finite floats (profit factor of a lossless group) become None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def main_cli(settings, args) -> None:
    import sys

    from heartless.core.store import Store
    from heartless.strategy.params import StrategyParams

    store = Store(settings.db_path)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else store.candle_symbols()
    years = getattr(args, "years", None)
    try:
        if years:
            if args.start or args.end:
                raise ValueError("--years and --start/--end are mutually exclusive")
            start, end, _ = year_window(years)
        elif args.start or args.end:
            sp = store_splits(store, symbols) if not (args.start and args.end) else {}
            start = parse_when(args.start) if args.start else sp["all"][0]
            end = parse_when(args.end, end=True) if args.end else sp["all"][1]
        else:
            start, end = store_splits(store, symbols)[args.split]
        if end <= start:
            raise ValueError(f"empty window {_iso(start)} .. {_iso(end)}")
        unseal = sorted(set(getattr(args, "unseal", None) or []))
        seal.check_window(start, end, unseal)
    except (ValueError, seal.SealedError) as e:
        store.close()
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    argv = list(getattr(args, "argv", None) or sys.argv)
    touched = seal.overlapping(start, end)
    for period in unseal:
        if period in touched:
            print(f"\n{'!' * 78}\n!!! UNSEALING {period}: this run evaluates the sealed period {period}. It is the "
                  f"one-time final exam\n!!! (docs/ROADMAP.md section 1); re-tuning after seeing it is forbidden. "
                  f"Recorded in\n!!! {seal.LEDGER}\n{'!' * 78}\n", file=sys.stderr, flush=True)
            seal.record_unseal(period, argv)
        else:
            print(f"note: --unseal {period} has no effect: the window does not overlap it (not recorded)", file=sys.stderr)
    if args.params:
        with open(args.params) as f:
            params = StrategyParams.from_dict(json.load(f))
    else:
        rows = store.load_params_versions(role="champion", limit=1)
        params = StrategyParams.default()
        if rows:
            params, reset = StrategyParams.from_stored(rows[0]["params"])
            if reset:
                print(f"note: stored champion {rows[0]['id']} predates the current design of {', '.join(reset)}; "
                      f"shipped defaults are used for them")
    store.close()
    notes: list[dict] = []
    params = apply_overrides(params, args.set, notes)
    extra_settings = dict(STRESS_SETTINGS) if getattr(args, "stress", False) else {}
    if getattr(args, "meta", False):
        extra_settings["META_LABEL"] = True
    chunk_days = getattr(args, "chunk_days", None)
    res = evaluate(params, symbols, start, end, str(settings.db_path), only_alpha=args.alpha, workers=args.workers,
                   settings=extra_settings, chunk_days=chunk_days, unseal=[p for p in unseal if p in touched])
    if args.json:
        print(json.dumps(_clean({"summary": res.summary(), "summary_excl_boundary": res.summary_excl_boundary(),
                                 "by_year": res.by_year(), "by_alpha": res.by_alpha, "by_symbol": res.by_symbol,
                                 "by_month": res.by_month, "by_side": res.by_side(), "by_year_side": res.by_year_side()}),
                         default=str))
    else:
        print(res.report(by_side=getattr(args, "by_side", False)))
    if args.trades_out:
        with open(args.trades_out, "w") as f:
            json.dump(res.trades, f, default=str)
    if getattr(args, "out", None):
        doc = {**res.to_json(), "params_version": params.version, "alpha": args.alpha, "symbols": symbols,
               "overrides": list(args.set or []), "override_adjustments": notes, "settings_overrides": extra_settings,
               "chunk_days": chunk_days, "workers": args.workers, "git_commit": seal.git_commit(),
               "seal": seal.status(start, end, unseal), "argv": argv,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(_clean(doc), f, default=str, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)

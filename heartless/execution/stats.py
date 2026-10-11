"""Performance statistics over closed trades and equity curves."""
from __future__ import annotations

import math
from collections import defaultdict


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
                "profit_factor": 0.0, "expectancy": 0.0, "avg_r": 0.0, "std_r": 0.0, "t_stat": 0.0, "max_dd": 0.0,
                "max_dd_pct": 0.0, "fees": 0.0, "funding": 0.0, "avg_hold_min": 0.0, "best": 0.0, "worst": 0.0,
                "sharpe_like": 0.0}
    pnls = [float(t["pnl"]) for t in trades]
    rs = [float(t.get("r_multiple") or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw = sum(wins)
    gl = -sum(losses)
    net = sum(pnls)
    mean_r = sum(rs) / n
    var_r = sum((r - mean_r) ** 2 for r in rs) / max(n - 1, 1)
    std_r = math.sqrt(var_r)
    t_stat = mean_r / (std_r / math.sqrt(n)) if std_r > 0 else (mean_r * math.sqrt(n) * 10 if mean_r else 0.0)
    # drawdown on cumulative pnl ordered by exit time
    ordered = sorted(trades, key=lambda t: t.get("exit_time") or 0)
    cum = peak = 0.0
    max_dd = 0.0
    for t in ordered:
        cum += float(t["pnl"])
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    holds = [((t.get("exit_time") or 0) - (t.get("entry_time") or 0)) / 60000 for t in trades]
    return {"n": n, "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / n * 100, "net": net,
            "gross_win": gw, "gross_loss": gl, "profit_factor": (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
            "expectancy": net / n, "avg_r": mean_r, "std_r": std_r, "t_stat": t_stat, "max_dd": max_dd,
            "fees": sum(float(t.get("fees") or 0) for t in trades), "funding": sum(float(t.get("funding") or 0) for t in trades),
            "avg_hold_min": sum(holds) / n, "best": max(pnls), "worst": min(pnls),
            "sharpe_like": mean_r / std_r * math.sqrt(min(n, 252)) if std_r > 0 else 0.0}


def by_alpha(trades: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[t.get("alpha") or "?"].append(t)
    return {k: summarize(v) for k, v in groups.items()}


def by_symbol(trades: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[t.get("symbol") or "?"].append(t)
    return {k: summarize(v) for k, v in groups.items()}


def equity_drawdown(curve: list[tuple[int, float, float]]) -> tuple[float, float]:
    """Max drawdown (abs, pct) of an equity curve [(ts, balance, equity)]."""
    peak = -1.0
    dd = dd_pct = 0.0
    for _, _, eq in curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = max(dd, peak - eq)
            dd_pct = max(dd_pct, (peak - eq) / peak * 100)
    return dd, dd_pct


def objective(stats: dict, min_trades: int = 15, t_key: str = "t_stat") -> float:
    """Research objective: t-stat of R multiples, penalised by drawdown and rewarded by expectancy. `t_key` picks the
    t statistic ("t_day" = day-clustered, used by the walk-forward fold confirmation)."""
    n = stats.get("n", 0)
    if n < min_trades:
        return -1e9
    t = stats.get(t_key, stats["t_stat"])
    pf = min(stats["profit_factor"], 5.0)
    dd_pen = stats["max_dd"] / max(abs(stats["net"]) + 1e-9, 1.0) if stats["net"] > 0 else 2.0
    return t + 0.5 * (pf - 1.0) + 2.0 * stats["avg_r"] - 0.5 * dd_pen


def sparkline(values: list[float], width: int = 24) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return bars[3] * len(values)
    return "".join(bars[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in values)


# --- day-clustered statistics (ROADMAP P4) -----------------------------------------------------------
# Trades on correlated coins that close on the same day are not independent evidence: a market-wide move stops or
# lifts most of them together. Every promotion, retirement and fold-confirmation statistic is therefore computed on
# UTC exit days (the effective sample size is the number of trading days, not trades), like the lab's t_day.
MS_DAY_UTC = 86_400_000


def _r(t: dict) -> float:
    return float(t.get("r_multiple") or 0.0)


def exit_day(t: dict) -> int:
    """UTC day index (days since the epoch) on which a trade closed."""
    return int((t.get("exit_time") or 0) // MS_DAY_UTC)


def exit_days(trades: list[dict]) -> int:
    """Number of distinct UTC exit days."""
    return len({exit_day(t) for t in trades})


def daily_r(trades: list[dict]) -> dict[int, float]:
    """Sum of R per UTC exit day (day index -> R), days without a closed trade omitted."""
    out: dict[int, float] = defaultdict(float)
    for t in trades:
        out[exit_day(t)] += _r(t)
    return dict(sorted(out.items()))


def day_clustered_t(trades: list[dict], min_days: int = 5) -> float:
    """t of the mean R per trade with trades clustered by UTC exit day (cluster-robust variance), the same estimator
    as the lab's t_day (heartless/learning/lab.py). 0.0 with fewer than 3 trades or `min_days` exit days."""
    n = len(trades)
    if n < 3:
        return 0.0
    days = daily_r(trades)
    k = len(days)
    if k < min_days or k < 2:
        return 0.0
    mean = sum(_r(t) for t in trades) / n
    counts: dict[int, int] = defaultdict(int)
    for t in trades:
        counts[exit_day(t)] += 1
    var = sum((s - counts[d] * mean) ** 2 for d, s in days.items()) * k / (k - 1) / (n ** 2)
    return mean / math.sqrt(var) if var > 0 else 0.0


def paired_daily_diff(a: list[dict], b: list[dict], start_ms: int | None = None, end_ms: int | None = None) -> dict:
    """Paired comparison of two engines' daily R over the same calendar days: d_i = R_a(day i) - R_b(day i).

    The days are every UTC calendar day from `start_ms` to `end_ms` (both engines ran over the whole span, so a day on
    which one of them closed nothing is a real 0R day for it), or, without a span, the days on which either engine
    closed a trade. Returns days, mean difference per day, its t (0.0 with fewer than 2 days or no dispersion) and the
    R totals of both sides over those days."""
    ra, rb = daily_r(a), daily_r(b)
    if start_ms is not None and end_ms is not None and end_ms >= start_ms:
        days = list(range(int(start_ms // MS_DAY_UTC), int(end_ms // MS_DAY_UTC) + 1))
    else:
        days = sorted(set(ra) | set(rb))
    diffs = [ra.get(d, 0.0) - rb.get(d, 0.0) for d in days]
    k = len(diffs)
    out = {"days": k, "mean": 0.0, "t": 0.0, "sum_a": sum(ra.get(d, 0.0) for d in days),
           "sum_b": sum(rb.get(d, 0.0) for d in days)}
    if k == 0:
        return out
    mean = sum(diffs) / k
    out["mean"] = mean
    if k >= 2:
        sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (k - 1))
        out["t"] = mean / (sd / math.sqrt(k)) if sd > 0 else 0.0
    return out


def max_drawdown_r(trades: list[dict]) -> float:
    """Max drawdown of cumulative R ordered by exit time (scale-free: accounts of different size compare directly)."""
    cum = peak = dd = 0.0
    for t in sorted(trades, key=lambda t: t.get("exit_time") or 0):
        cum += _r(t)
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def profit_factor_r(trades: list[dict]) -> float:
    """Gross winning R / gross losing R (inf with wins and no losses, 0.0 without wins)."""
    win = sum(_r(t) for t in trades if _r(t) > 0)
    loss = -sum(_r(t) for t in trades if _r(t) < 0)
    return win / loss if loss > 0 else (float("inf") if win > 0 else 0.0)


def r_stats(trades: list[dict]) -> dict:
    """The R-unit record every promotion / retirement / kill-switch decision reads (and stores as its inputs)."""
    n = len(trades)
    return {"n": n, "days": exit_days(trades), "avg_r": (sum(_r(t) for t in trades) / n) if n else 0.0,
            "sum_r": sum(_r(t) for t in trades), "t_day": day_clustered_t(trades), "pf_r": profit_factor_r(trades),
            "max_dd_r": max_drawdown_r(trades)}


def cusum_lower(rs: list[float], expected: float = 0.0, k: float = 0.25, scale: float = 1.0) -> tuple[float, float]:
    """One-sided lower CUSUM of R multiples: S_t = min(0, S_{t-1} + (R_t - expected) / scale + k). Returns (final S,
    lowest S), in units of `scale`. `k` is the slack per trade: a trade that loses less than k below expectation pulls S
    back toward zero, so only a sustained shortfall accumulates. With `scale` = the R standard deviation, k and the alarm
    line are in standard deviations, which keeps the false-alarm rate independent of how dispersed an alpha's R is."""
    s = low = 0.0
    sc = scale if scale and scale > 0 else 1.0
    for r in rs:
        s = min(0.0, s + (float(r) - expected) / sc + k)
        low = min(low, s)
    return s, low


# --- execution quality (ROADMAP P4) ------------------------------------------------------------------
def slippage_bps(order_side: str, price: float, ref: float) -> float:
    """Signed slippage of a fill against its reference price in basis points; positive = cost (a buy filled above,
    a sell below the reference)."""
    if not ref or ref <= 0 or not price:
        return 0.0
    sign = 1.0 if order_side == "BUY" else -1.0
    return (price - ref) / ref * 1e4 * sign


def exec_quality(rows: list[dict], model_slippage_bps: float, min_fills: int = 50) -> dict:
    """Rolling execution-quality aggregate over fill rows (slip_bps, fee, model_fee, notional, maker).

    Slippage is averaged over taker fills only (market entries / closes and triggered stops / take-profits are what
    BACKTEST_SLIPPAGE_BPS models; a resting maker fill executes at its own price). `suggest` is set when at least
    `min_fills` taker fills average more than twice the modelled slippage; nothing is ever changed automatically."""
    taker = [r for r in rows if not r.get("maker")]
    slips = sorted(float(r.get("slip_bps") or 0.0) for r in taker)
    notional = sum(float(r.get("notional") or 0.0) for r in rows)
    fee = sum(float(r.get("fee") or 0.0) for r in rows)
    model_fee = sum(float(r.get("model_fee") or 0.0) for r in rows)
    out = {"fills": len(rows), "taker_fills": len(taker), "maker_fills": len(rows) - len(taker),
           "slip_bps": (sum(slips) / len(slips)) if slips else 0.0,
           "slip_p90_bps": slips[min(len(slips) - 1, int(0.9 * len(slips)))] if slips else 0.0,
           "fee_bps": fee / notional * 1e4 if notional > 0 else 0.0,
           "model_fee_bps": model_fee / notional * 1e4 if notional > 0 else 0.0,
           "fee_vs_model": (fee / model_fee) if model_fee > 0 else 0.0, "model_slip_bps": model_slippage_bps,
           "suggest": False}
    out["suggest"] = len(taker) >= min_fills and out["slip_bps"] > 2.0 * model_slippage_bps
    return out

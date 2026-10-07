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


def objective(stats: dict, min_trades: int = 15) -> float:
    """Research objective: t-stat of R multiples, penalised by drawdown and rewarded by expectancy."""
    n = stats.get("n", 0)
    if n < min_trades:
        return -1e9
    t = stats["t_stat"]
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

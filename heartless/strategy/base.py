from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from heartless.core.models import Regime, Signal, SymbolInfo, Ticker
from heartless.data.context import EMPTY
from heartless.data.features import MarketView


@dataclass
class Context:
    symbol: str
    info: SymbolInfo
    ticker: Ticker
    regime: Regime
    regime_info: dict = field(default_factory=dict)
    btc_regime: Regime | None = None
    oi_change: float | None = None  # fractional OI change over ~4h, None if unknown
    now: int = 0
    funding_rate: float = 0.0
    minutes_to_funding: float = 999.0
    # the symbol's current settlement interval in minutes (480 for most symbols; 240 / 120 / 60 while Binance settles
    # it more often), so "minutes since the last settlement" is funding_interval_min - minutes_to_funding
    funding_interval_min: float = 480.0
    book_imbalance: float = 0.0  # (bidQty-askQty)/(bidQty+askQty)
    # derivatives positioning snapshot (open interest, long/short ratios, taker ratio) - see data/extras.py.
    # Identical computation in live trading and backtests; empty when no metrics are available.
    extras: dict = field(default_factory=dict)
    # market context at the last hour close (data/context.py): the symbol's cross-sectional features (return ranks and
    # z-scores, volume share, funding, OI change, distance from highs/lows) plus market-wide ones (BTC returns and trend,
    # breadth, dispersion, median funding, aggregate OI) and "ts". Read-only and identical in live trading and backtests;
    # empty when no context is available, so alphas must treat a missing key (or NaN) as "unknown".
    market: Mapping = field(default_factory=lambda: EMPTY)


def _longest(attr: str, params_list, only_alpha: str | None) -> float:
    from heartless.strategy.alphas import ALL_ALPHAS
    from heartless.strategy.params import StrategyParams

    best = 0.0
    for p in params_list:
        if p is None:
            continue
        enabled = (StrategyParams.from_dict(p) if isinstance(p, Mapping) else p).enabled
        for a in ALL_ALPHAS:
            days = getattr(a, attr, None)
            if not days:
                continue
            if (a.name == only_alpha) if only_alpha else enabled.get(a.name, True):
                best = max(best, float(days))
    return best


def required_lookback_days(params_list, only_alpha: str | None = None) -> float:
    """Longest Alpha.lookback_days among the alphas evaluated under any of `params_list` (StrategyParams or their
    dicts, read like StrategyParams.from_dict): `only_alpha` alone when given (an isolation run evaluates it even when
    switched off), otherwise every enabled alpha as the ensemble sees it. 0.0 when none of them declares one."""
    return _longest("lookback_days", params_list, only_alpha)


def required_hold_days(params_list, only_alpha: str | None = None) -> float:
    """Longest Alpha.max_hold_days among the evaluated alphas (as required_lookback_days); 0.0 when none declares one."""
    return _longest("max_hold_days", params_list, only_alpha)


def ok(*vals: float) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


class Alpha(ABC):
    """One setup. evaluate() returns a Signal whose fields also tell the engine how to run the trade: entry style,
    time stop, trailing, and the opt-ins exit_on_regime_change (fades) and entry_ttl_bars (resting retest limits)."""
    name: str = "alpha"
    timeframe: str = "5m"
    description: str = ""
    param_specs: list = []  # list[ParamSpec]; min_conf and tp1_frac are added by the registry
    regime_affinity: dict | None = None  # {Regime: weight 0..1}
    enabled_by_default: bool = True
    # Days of history the alpha's indicators need before a decision (e.g. 200 for a 200-day EMA on view.tf("1d")).
    # None: the defaults cover it (lab.WARMUP_MS = 12 days of warm-up, the live 1m history of HISTORY_DAYS /
    # LIVE_HISTORY_DAYS). A value extends the lab / backtester warm-up and the live 1m history (memory, backfill and
    # the daily trim) whenever the alpha is evaluated, or enabled in the champion or a challenger
    # (required_lookback_days).
    lookback_days: float | None = None
    # Longest a position of this alpha can stay open, in days (the upper bound of its hold parameter). None: within
    # the lab's chunk hand-off (lab.BURN_IN_MS / RUNOFF_MS = 7 days; the longest shipped time stop is 96 h). A value
    # lengthens the burn-in and run-off of chunked lab runs, so a multi-week trade is neither cut at a chunk end nor
    # missed by the next chunk's re-creation of the positions held across the cut.
    max_hold_days: float | None = None

    @abstractmethod
    def evaluate(self, view: MarketView, ctx: Context, p: dict[str, float]) -> Signal | None: ...

    @staticmethod
    def bars_to_1m(n: int, tf: str) -> int:
        from heartless.util.timeutil import TF_MS

        return int(n * TF_MS[tf] // TF_MS["1m"])

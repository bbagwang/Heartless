from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from heartless.core.models import Regime, Signal, SymbolInfo, Ticker
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
    book_imbalance: float = 0.0  # (bidQty-askQty)/(bidQty+askQty)
    # derivatives positioning snapshot (open interest, long/short ratios, taker ratio) - see data/extras.py.
    # Identical computation in live trading and backtests; empty when no metrics are available.
    extras: dict = field(default_factory=dict)


def ok(*vals: float) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


class Alpha(ABC):
    name: str = "alpha"
    timeframe: str = "5m"
    description: str = ""
    param_specs: list = []  # list[ParamSpec]; min_conf and tp1_frac are added by the registry
    regime_affinity: dict | None = None  # {Regime: weight 0..1}
    enabled_by_default: bool = True

    @abstractmethod
    def evaluate(self, view: MarketView, ctx: Context, p: dict[str, float]) -> Signal | None: ...

    @staticmethod
    def bars_to_1m(n: int, tf: str) -> int:
        from heartless.util.timeutil import TF_MS

        return int(n * TF_MS[tf] // TF_MS["1m"])

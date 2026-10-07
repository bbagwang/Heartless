"""Abstract account interface implemented by the paper simulator and the live Binance account."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from heartless.core.models import AccountState, Fill, SymbolInfo


@dataclass(slots=True)
class OrderResult:
    order_id: str = ""
    client_id: str = ""
    status: str = "NEW"  # NEW | FILLED | PARTIALLY_FILLED | CANCELED | REJECTED | EXPIRED
    filled_qty: float = 0.0
    avg_price: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.status == "FILLED"


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    qty: float  # signed: >0 long, <0 short
    entry_price: float
    unrealized: float = 0.0
    leverage: int = 0
    mark: float = 0.0


FillHandler = Callable[[Fill], Awaitable[None]]


class Account(ABC):
    name: str = "account"
    is_paper: bool = True

    def __init__(self) -> None:
        self._fill_handlers: list[FillHandler] = []
        self.symbols: dict[str, SymbolInfo] = {}

    def on_fill(self, handler: FillHandler) -> None:
        self._fill_handlers.append(handler)

    async def _emit_fill(self, fill: Fill) -> None:
        for h in list(self._fill_handlers):
            await h(fill)

    def set_symbols(self, symbols: dict[str, SymbolInfo]) -> None:
        self.symbols = symbols

    # --- lifecycle -----------------------------------------------------------------------------
    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def stop(self) -> None:  # pragma: no cover - trivial
        return None

    # --- queries -------------------------------------------------------------------------------
    @abstractmethod
    async def get_state(self) -> AccountState: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, PositionSnapshot]: ...

    @abstractmethod
    async def open_orders(self, symbol: str | None = None) -> list[dict]: ...

    @abstractmethod
    async def open_algo_orders(self, symbol: str | None = None) -> list[dict]: ...

    # --- trading -------------------------------------------------------------------------------
    @abstractmethod
    async def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False,
                           client_id: str = "") -> OrderResult: ...

    @abstractmethod
    async def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                          reduce_only: bool = False, client_id: str = "") -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str = "", client_id: str = "") -> bool: ...

    @abstractmethod
    async def query_order(self, symbol: str, order_id: str = "", client_id: str = "") -> OrderResult: ...

    @abstractmethod
    async def place_stop(self, symbol: str, side: str, trigger_price: float, qty: float | None = None,
                         close_position: bool = False, client_id: str = "") -> str: ...

    @abstractmethod
    async def place_take_profit(self, symbol: str, side: str, trigger_price: float, qty: float,
                                client_id: str = "") -> str: ...

    @abstractmethod
    async def cancel_algo(self, symbol: str, algo_id: str) -> bool: ...

    @abstractmethod
    async def cancel_all(self, symbol: str) -> None: ...

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        return None

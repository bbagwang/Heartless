"""Core data structures shared by live, paper and backtest code paths."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def order_side(self) -> str:
        """Binance order side that OPENS this position side."""
        return "BUY" if self is Side.LONG else "SELL"

    @property
    def close_order_side(self) -> str:
        return "SELL" if self is Side.LONG else "BUY"

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


class PositionStatus(str, Enum):
    PENDING = "PENDING"  # entry order working, not filled yet
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"  # entry never filled


class EntryStyle(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"  # post-only at touch, converts/cancels after validity


@dataclass(slots=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float
    close_time: int
    closed: bool = True

    @classmethod
    def from_rest(cls, row: list) -> "Candle":
        return cls(
            open_time=int(row[0]), open=float(row[1]), high=float(row[2]), low=float(row[3]),
            close=float(row[4]), volume=float(row[5]), close_time=int(row[6]), quote_volume=float(row[7]),
            trades=int(row[8]), taker_buy_volume=float(row[9]), closed=True,
        )

    @classmethod
    def from_ws(cls, k: dict) -> "Candle":
        return cls(
            open_time=int(k["t"]), open=float(k["o"]), high=float(k["h"]), low=float(k["l"]), close=float(k["c"]),
            volume=float(k["v"]), close_time=int(k["T"]), quote_volume=float(k["q"]), trades=int(k["n"]),
            taker_buy_volume=float(k["V"]), closed=bool(k["x"]),
        )


@dataclass(slots=True)
class SymbolInfo:
    symbol: str
    base: str
    quote: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float
    price_precision: int
    quantity_precision: int
    max_leverage: int = 20
    max_algo_orders: int = 10
    status: str = "TRADING"
    contract_type: str = "PERPETUAL"
    onboard_date: int = 0


@dataclass(slots=True)
class Ticker:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    mark: float = 0.0
    last: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: int = 0
    index: float = 0.0
    ts: int = 0

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.mark or self.last

    @property
    def ref(self) -> float:
        """Best available reference price."""
        return self.mark or self.mid or self.last

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 1e4 if m and self.bid and self.ask else 0.0


@dataclass(slots=True)
class Signal:
    """Output of a single alpha for one symbol at one bar."""
    alpha: str
    symbol: str
    side: Side
    confidence: float  # 0..1
    reason: str
    stop: float  # absolute stop price
    take_profit: float | None  # absolute TP for the runner (None => trail only)
    tp1: float | None  # partial take profit price (None => no partial)
    entry_style: EntryStyle = EntryStyle.MARKET
    limit_price: float | None = None
    max_hold_bars: int = 0  # in base bars (1m) ; 0 => no time stop
    trail_atr_mult: float = 0.0  # 0 => no trailing
    atr: float = 0.0
    timeframe: str = "5m"
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def r_distance(self) -> float:
        ref = self.limit_price if self.limit_price else self.tags.get("ref_price", 0.0)
        return abs(ref - self.stop) if ref else 0.0


@dataclass(slots=True)
class Decision:
    """Ensemble decision: possibly a merged view of several agreeing alphas."""
    symbol: str
    side: Side
    score: float
    confidence: float
    alphas: list[str]
    primary: Signal
    reason: str
    regime: Regime
    size_mult: float = 1.0
    expected_r: float = 0.0


@dataclass(slots=True)
class Fill:
    symbol: str
    order_side: str  # BUY / SELL
    qty: float
    price: float
    fee: float
    ts: int
    client_id: str = ""
    order_id: str = ""
    reduce_only: bool = False
    maker: bool = False
    kind: str = "ENTRY"  # ENTRY | SL | TP | CLOSE | UNKNOWN


@dataclass
class Position:
    id: str
    engine: str
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_time: int
    stop: float
    take_profit: float | None
    tp1: float | None
    initial_stop: float
    alpha: str
    alphas: list[str]
    reason: str
    confidence: float
    regime: str
    risk_amount: float
    r_unit: float
    notional: float
    leverage: int
    params_version: str
    atr: float = 0.0
    trail_atr_mult: float = 0.0
    max_hold_bars: int = 0
    timeframe: str = "5m"
    status: PositionStatus = PositionStatus.PENDING
    tp1_done: bool = False
    be_moved: bool = False
    fees: float = 0.0
    funding: float = 0.0
    realized: float = 0.0
    max_fav: float = 0.0  # max favourable excursion in price terms
    max_adv: float = 0.0
    exit_price: float | None = None
    exit_time: int | None = None
    exit_reason: str = ""
    r_multiple: float | None = None
    bars_held: int = 0
    entry_client_id: str = ""
    entry_order_id: str = ""
    sl_algo_id: str = ""
    tp_algo_id: str = ""
    original_qty: float = 0.0
    filled_qty: float = 0.0
    entry_style: str = "MARKET"
    limit_price: float | None = None
    pending_since: int = 0
    requotes: int = 0
    expected_profit: float = 0.0
    last_stop_update: int = 0
    stop_dirty: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # --- helpers -----------------------------------------------------------------------------
    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty * self.side.sign

    def pnl_pct(self, price: float) -> float:
        if not self.entry_price:
            return 0.0
        return (price - self.entry_price) / self.entry_price * 100 * self.side.sign

    def r_now(self, price: float) -> float:
        if not self.r_unit:
            return 0.0
        return (price - self.entry_price) * self.side.sign / self.r_unit

    def net_pnl(self) -> float:
        return self.realized - self.fees + self.funding

    def to_row(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        d["alphas"] = json.dumps(self.alphas)
        d["extra"] = json.dumps(self.extra, default=str)
        return d

    @classmethod
    def from_row(cls, row: dict) -> "Position":
        row = dict(row)
        row["side"] = Side(row["side"])
        row["status"] = PositionStatus(row["status"])
        row["alphas"] = json.loads(row.get("alphas") or "[]")
        row["extra"] = json.loads(row.get("extra") or "{}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})


@dataclass(slots=True)
class TradeRecord:
    """Closed trade summary used for learning and reporting."""
    position_id: str
    engine: str
    symbol: str
    side: str
    alpha: str
    alphas: list[str]
    regime: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    pnl: float  # net of fees and funding
    gross: float
    fees: float
    funding: float
    r_multiple: float
    risk_amount: float
    exit_reason: str
    confidence: float
    params_version: str
    bars_held: int
    max_fav_r: float
    max_adv_r: float
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alphas"] = json.dumps(self.alphas)
        return d


@dataclass(slots=True)
class AccountState:
    balance: float  # wallet balance
    equity: float  # balance + unrealized
    available: float
    unrealized: float
    ts: int

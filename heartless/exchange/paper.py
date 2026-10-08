"""Paper / simulation account.

Simulates Binance USDⓈ-M futures execution with realistic frictions:
  * taker/maker fees, slippage proportional to order size, post-only rejection when crossing,
  * conditional (algo) orders triggered on mark price, pessimistic intrabar ordering in backtests
    (the extreme adverse to the open position is visited first; conventional O-L-H-C / O-H-L-C when flat),
  * resting limit orders fill only when the opposite side of the book trades through them,
  * funding payments at every settlement (bar mode: driven by the backtester; tick mode: on the
    next_funding_time rollover of the markPrice stream), isolated-margin style equity accounting.
The same class powers live paper trading (tick-driven) and the backtester (bar-driven).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from heartless.core.models import AccountState, Candle, Fill, Ticker
from heartless.exchange.base import Account, OrderResult, PositionSnapshot
from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PaperPosition:
    qty: float = 0.0  # signed
    entry: float = 0.0
    leverage: int = 10

    @property
    def side(self) -> str:
        return "LONG" if self.qty > 0 else "SHORT" if self.qty < 0 else "FLAT"


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    client_id: str
    symbol: str
    side: str
    qty: float
    price: float | None  # None => market
    reduce_only: bool
    post_only: bool
    status: str = "NEW"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    created: int = 0


@dataclass(slots=True)
class PaperAlgo:
    algo_id: str
    client_id: str
    symbol: str
    side: str
    kind: str  # STOP_MARKET | TAKE_PROFIT_MARKET
    trigger: float
    qty: float | None
    close_position: bool
    status: str = "NEW"
    created: int = 0


class PaperAccount(Account):
    is_paper = True

    def __init__(self, name: str = "paper", initial_balance: float = 10_000.0, taker_fee: float = 0.0005,
                 maker_fee: float = 0.0002, slippage_bps: float = 1.5, impact_bps_per_10k: float = 0.3,
                 clock=None, spread_bps: float = 1.0):
        super().__init__()
        self.name = name
        self.initial_balance = initial_balance
        self.wallet = initial_balance
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_bps = slippage_bps
        self.impact_bps_per_10k = impact_bps_per_10k
        self.spread_bps = spread_bps
        self.positions: dict[str, PaperPosition] = {}
        self.orders: dict[str, PaperOrder] = {}  # working orders only (terminal ones move to _done_orders)
        self.algos: dict[str, PaperAlgo] = {}  # working algos only (terminal ones move to _done_algos)
        self._done_orders: dict[str, PaperOrder] = {}  # bounded history so query_order still answers
        self._done_algos: dict[str, PaperAlgo] = {}
        self.tickers: dict[str, Ticker] = {}
        self._ids = itertools.count(1)
        self._clock = clock or now_ms
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.realized = 0.0
        self.trade_count = 0
        self.leverages: dict[str, int] = {}
        # tick mode funding: per symbol (next settlement time, rate quoted for it) and the last settled time
        self._funding_next: dict[str, tuple[int, float]] = {}
        self._funding_settled: dict[str, int] = {}

    # --- helpers -------------------------------------------------------------------------------
    def now(self) -> int:
        return int(self._clock())

    def reset(self, balance: float | None = None) -> None:
        self.wallet = balance if balance is not None else self.initial_balance
        self.initial_balance = self.wallet
        self.positions.clear()
        self.orders.clear()
        self.algos.clear()
        self._done_orders.clear()
        self._done_algos.clear()
        self.fees_paid = self.funding_paid = self.realized = 0.0
        self.trade_count = 0
        self._funding_next.clear()
        self._funding_settled.clear()

    def _slip(self, symbol: str, notional: float) -> float:
        bps = self.slippage_bps + self.impact_bps_per_10k * (notional / 10_000.0)
        return min(bps, 25.0) / 1e4

    def _ref(self, symbol: str, side: str) -> float:
        t = self.tickers.get(symbol)
        if not t:
            return 0.0
        if side == "BUY":
            return t.ask or t.mark or t.last or t.bid
        return t.bid or t.mark or t.last or t.ask

    def mark(self, symbol: str) -> float:
        t = self.tickers.get(symbol)
        return (t.mark or t.mid or t.last) if t else 0.0

    _DONE_KEEP = 500

    def _prune(self) -> None:
        """Move terminal orders/algos out of the working sets. Without this every trigger check walks every order
        ever placed, which makes long backtests quadratic."""
        for oid in [k for k, o in self.orders.items() if o.status not in ("NEW", "PARTIALLY_FILLED")]:
            self._done_orders[oid] = self.orders.pop(oid)
        for aid in [k for k, a in self.algos.items() if a.status != "NEW"]:
            self._done_algos[aid] = self.algos.pop(aid)
        for done in (self._done_orders, self._done_algos):
            while len(done) > self._DONE_KEEP:
                done.pop(next(iter(done)))

    def has_working(self, symbol: str) -> bool:
        return any(o.symbol == symbol for o in self.orders.values()) or any(a.symbol == symbol for a in self.algos.values())

    def unrealized_total(self) -> float:
        tot = 0.0
        for s, p in self.positions.items():
            if p.qty:
                m = self.mark(s)
                if m:
                    tot += (m - p.entry) * p.qty
        return tot

    def equity(self) -> float:
        return self.wallet + self.unrealized_total()

    # --- Account API ---------------------------------------------------------------------------
    async def get_state(self) -> AccountState:
        unreal = self.unrealized_total()
        used = sum(abs(p.qty) * self.mark(s) / max(p.leverage, 1) for s, p in self.positions.items() if p.qty)
        return AccountState(balance=self.wallet, equity=self.wallet + unreal,
                            available=max(self.wallet + unreal - used, 0.0), unrealized=unreal, ts=self.now())

    async def get_positions(self) -> dict[str, PositionSnapshot]:
        out = {}
        for s, p in self.positions.items():
            if abs(p.qty) > 0:
                m = self.mark(s)
                out[s] = PositionSnapshot(symbol=s, qty=p.qty, entry_price=p.entry, unrealized=(m - p.entry) * p.qty if m else 0.0,
                                          leverage=p.leverage, mark=m)
        return out

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        return [vars(o) if not hasattr(o, "__slots__") else {k: getattr(o, k) for k in o.__slots__}
                for o in self.orders.values() if o.status in ("NEW", "PARTIALLY_FILLED") and (symbol is None or o.symbol == symbol)]

    async def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        return [{k: getattr(a, k) for k in a.__slots__} for a in self.algos.values()
                if a.status == "NEW" and (symbol is None or a.symbol == symbol)]

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        self.leverages[symbol] = leverage
        if symbol in self.positions:
            self.positions[symbol].leverage = leverage

    async def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False,
                           client_id: str = "") -> OrderResult:
        oid = str(next(self._ids))
        ref = self._ref(symbol, side)
        if ref <= 0:
            return OrderResult(oid, client_id, "REJECTED", raw={"msg": "no price"})
        if reduce_only:
            pos = self.positions.get(symbol)
            if not pos or pos.qty == 0 or (pos.qty > 0 and side == "BUY") or (pos.qty < 0 and side == "SELL"):
                return OrderResult(oid, client_id, "REJECTED", raw={"msg": "ReduceOnly Order is rejected", "code": -2022})
            qty = min(qty, abs(pos.qty))
        notional = qty * ref
        slip = self._slip(symbol, notional)
        px = ref * (1 + slip) if side == "BUY" else ref * (1 - slip)
        fill = await self._execute(symbol, side, qty, px, maker=False, client_id=client_id, order_id=oid, reduce_only=reduce_only)
        return OrderResult(oid, client_id, "FILLED", qty, fill.price, raw={})

    async def limit_order(self, symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                          reduce_only: bool = False, client_id: str = "") -> OrderResult:
        oid = str(next(self._ids))
        t = self.tickers.get(symbol)
        if t and post_only:
            if (side == "BUY" and t.ask and price >= t.ask) or (side == "SELL" and t.bid and price <= t.bid):
                # GTX order that would cross is rejected/expired by Binance
                return OrderResult(oid, client_id, "EXPIRED", raw={"msg": "post-only would cross"})
        o = PaperOrder(oid, client_id, symbol, side, qty, price, reduce_only, post_only, created=self.now())
        self.orders[oid] = o
        if not post_only and t:  # a crossing GTC limit fills immediately as taker
            if (side == "BUY" and t.ask and price >= t.ask) or (side == "SELL" and t.bid and price <= t.bid):
                await self._fill_order(o, t.ask if side == "BUY" else t.bid, maker=False)
        return OrderResult(oid, client_id, o.status, o.filled_qty, o.avg_price)

    async def cancel_order(self, symbol: str, order_id: str = "", client_id: str = "") -> bool:
        for o in list(self.orders.values()) + list(self._done_orders.values()):
            if o.symbol == symbol and (o.order_id == order_id or (client_id and o.client_id == client_id)):
                if o.status in ("NEW", "PARTIALLY_FILLED"):
                    o.status = "CANCELED"
                    self._prune()
                    return True
                return False
        return False

    async def query_order(self, symbol: str, order_id: str = "", client_id: str = "") -> OrderResult:
        for o in list(self.orders.values()) + list(reversed(list(self._done_orders.values()))):
            if o.symbol == symbol and (o.order_id == order_id or (client_id and o.client_id == client_id)):
                return OrderResult(o.order_id, o.client_id, o.status, o.filled_qty, o.avg_price)
        return OrderResult(order_id, client_id, "UNKNOWN")

    async def place_stop(self, symbol: str, side: str, trigger_price: float, qty: float | None = None,
                         close_position: bool = False, client_id: str = "") -> str:
        aid = "A" + str(next(self._ids))
        self.algos[aid] = PaperAlgo(aid, client_id, symbol, side, "STOP_MARKET", trigger_price, qty, close_position,
                                    created=self.now())
        return aid

    async def place_take_profit(self, symbol: str, side: str, trigger_price: float, qty: float,
                                client_id: str = "") -> str:
        aid = "A" + str(next(self._ids))
        self.algos[aid] = PaperAlgo(aid, client_id, symbol, side, "TAKE_PROFIT_MARKET", trigger_price, qty, False,
                                    created=self.now())
        return aid

    async def cancel_algo(self, symbol: str, algo_id: str) -> bool:
        a = self.algos.get(algo_id)
        if a and a.symbol == symbol and a.status == "NEW":  # ids restart per process: never touch another symbol's bracket
            a.status = "CANCELED"
            self._prune()
            return True
        return False

    async def cancel_all(self, symbol: str) -> None:
        for o in self.orders.values():
            if o.symbol == symbol and o.status in ("NEW", "PARTIALLY_FILLED"):
                o.status = "CANCELED"
        for a in self.algos.values():
            if a.symbol == symbol and a.status == "NEW":
                a.status = "CANCELED"
        self._prune()

    # --- simulation drivers --------------------------------------------------------------------
    async def on_ticker(self, t: Ticker) -> None:
        """Live paper mode: called on every book/mark update."""
        self.tickers[t.symbol] = t
        await self._settle_funding(t)
        await self._check_triggers(t.symbol, t.mark or t.mid or t.last, t.bid, t.ask, t.ts)

    async def _settle_funding(self, t: Ticker) -> None:
        """Tick mode: pay/receive funding once the settlement time announced by the markPrice stream has passed.

        The stream's `r`/`T` describe the NEXT settlement and `r` flips to the following period's rate right
        after `T`, so the pair is cached from earlier ticks and the cached rate is applied when a later tick's
        timestamp crosses the cached `T`. Each settlement time is applied at most once per symbol. Bar mode
        (backtester) settles funding itself through apply_funding and never reaches this path.
        """
        sym = t.symbol
        pending = self._funding_next.get(sym)
        if pending and t.ts >= pending[0]:
            due, rate = pending
            del self._funding_next[sym]
            self._funding_settled[sym] = due
            if rate:
                await self.apply_funding(sym, rate, t.mark or t.mid or t.last, due)
        nft = t.next_funding_time
        if nft and nft != self._funding_settled.get(sym):
            self._funding_next[sym] = (nft, t.funding_rate)

    async def on_bar(self, symbol: str, c: Candle) -> None:
        """Backtest mode: process a completed bar.

        Price path: open -> first extreme -> second extreme -> close. For an open position the ADVERSE extreme
        is visited first (long: low then high, short: high then low), so a stop is always evaluated before a
        take profit that sits in the same bar. When flat the conventional path is used (bullish bar: low then
        high, bearish: high then low), so a resting entry fills at the last-visited extreme and cannot exit in
        the same bar. Algos hit between two price points fill at their trigger; algos already through their
        trigger at a price point (open/close, or placed by a fill handler at that point) fill at the market.
        """
        t = self.tickers.get(symbol) or Ticker(symbol)
        half = self.spread_bps / 2e4
        if not self.has_working(symbol):
            # nothing can trigger or fill: just publish the closing book (dominant case in backtests)
            t.mark = t.last = c.close
            t.bid = c.close * (1 - half)
            t.ask = c.close * (1 + half)
            t.ts = c.close_time
            self.tickers[symbol] = t
            return

        def set_px(px: float) -> None:
            t.mark = t.last = px
            t.bid = px * (1 - half)
            t.ask = px * (1 + half)

        # 1) at bar open, pending limit orders / triggers may fill
        set_px(c.open)
        t.ts = c.open_time
        self.tickers[symbol] = t
        await self._check_triggers(symbol, c.open, t.bid, t.ask, c.open_time)
        # 2) intrabar: adverse extreme first for an open position (pessimistic: stop before target);
        #    flat -> conventional path (bullish O-L-H-C, bearish O-H-L-C)
        pos = self.positions.get(symbol)
        if pos and pos.qty > 0:
            first, second = c.low, c.high
        elif pos and pos.qty < 0:
            first, second = c.high, c.low
        else:
            first, second = (c.high, c.low) if c.close < c.open else (c.low, c.high)
        for px in (first, second):
            set_px(px)
            await self._check_triggers(symbol, px, t.bid, t.ask, c.open_time, intrabar=True)
        # 3) settle at close
        set_px(c.close)
        t.ts = c.close_time
        await self._check_triggers(symbol, c.close, t.bid, t.ask, c.close_time)

    async def apply_funding(self, symbol: str, rate: float, mark: float, ts: int) -> None:
        p = self.positions.get(symbol)
        if not p or p.qty == 0 or not mark:
            return
        payment = -p.qty * mark * rate  # long pays when rate>0
        self.wallet += payment
        self.funding_paid += payment
        await self._emit_fill(Fill(symbol, "FUNDING", 0.0, mark, -payment, ts, kind="FUNDING"))

    async def _check_triggers(self, symbol: str, mark: float, bid: float, ask: float, ts: int,
                              intrabar: bool = False) -> None:
        if not mark:
            return
        passes = 0
        while True:
            known = set(self.algos)
            await self._check_triggers_once(symbol, mark, bid, ask, ts, intrabar)
            passes += 1
            # Fill handlers (the engine) may have placed algos during this pass, e.g. the brackets of an entry
            # that has just filled. They were created with the market already AT this price point, so one
            # that is already through its trigger fires now, at the market (Binance rejects such an order
            # with -2021 and the engine's software backstop exits at market) -- never later at a trigger
            # price the bar did not revisit. Hence evaluate them here, non-intrabar.
            intrabar = False
            if passes >= 8 or not any(k not in known and a.status == "NEW" and a.symbol == symbol
                                      for k, a in self.algos.items()):
                break
        self._prune()

    async def _check_triggers_once(self, symbol: str, mark: float, bid: float, ask: float, ts: int,
                                   intrabar: bool) -> None:
        # algo orders (stops first, then take profits) -------------------------------------------
        for a in sorted(self.algos.values(), key=lambda x: 0 if x.kind == "STOP_MARKET" else 1):
            if a.symbol != symbol or a.status != "NEW":
                continue
            hit = False
            if a.kind == "STOP_MARKET":
                hit = (a.side == "SELL" and mark <= a.trigger) or (a.side == "BUY" and mark >= a.trigger)
            else:
                hit = (a.side == "SELL" and mark >= a.trigger) or (a.side == "BUY" and mark <= a.trigger)
            if not hit:
                continue
            a.status = "TRIGGERED"
            pos = self.positions.get(symbol)
            if not pos or pos.qty == 0:
                a.status = "EXPIRED"
                continue
            if (a.side == "SELL" and pos.qty <= 0) or (a.side == "BUY" and pos.qty >= 0):
                a.status = "EXPIRED"
                continue
            qty = abs(pos.qty) if a.close_position or a.qty is None else min(a.qty, abs(pos.qty))
            # intrabar: the move went through the trigger -> fill at the trigger; otherwise (open/close
            # evaluation, or an algo placed at this price point) the price already sits beyond the trigger ->
            # the market order fills on the BOOK side it hits (bid for SELL, ask for BUY; the mark price that
            # triggers it is index-based and need not be executable), stops never better than their trigger
            if intrabar:
                base = a.trigger
            else:
                book = (bid if a.side == "SELL" else ask) or mark
                if a.kind == "STOP_MARKET":
                    base = min(book, a.trigger) if a.side == "SELL" else max(book, a.trigger)
                else:
                    base = book
            slip = self._slip(symbol, qty * base)
            px = base * (1 - slip) if a.side == "SELL" else base * (1 + slip)
            kind = "SL" if a.kind == "STOP_MARKET" else "TP"
            await self._execute(symbol, a.side, qty, px, maker=False, client_id=a.client_id, order_id=a.algo_id,
                                reduce_only=True, kind=kind, ts=ts)
            a.status = "FINISHED"
        # resting limit orders -----------------------------------------------------------------
        # A maker order fills when the OPPOSITE side of the book trades through it (BUY: best ask at or below
        # our bid, SELL: best bid at or above our offer), not when the index-based mark merely touches it.
        # In bar mode the book is synthesized around each price point (ask = px * (1 + half spread)), so the
        # bar's extreme has to trade through the limit by half the spread; a low that only equals the limit
        # leaves the order resting. Without a book (mark-only tick) the mark must trade strictly through.
        for o in list(self.orders.values()):
            if o.symbol != symbol or o.status not in ("NEW", "PARTIALLY_FILLED") or o.price is None:
                continue
            if o.side == "BUY":
                hit = ask <= o.price if ask else mark < o.price
            else:
                hit = bid >= o.price if bid else mark > o.price
            if hit:
                await self._fill_order(o, o.price, maker=True, ts=ts)

    async def _fill_order(self, o: PaperOrder, px: float, maker: bool, ts: int | None = None) -> None:
        qty = o.qty - o.filled_qty
        if o.reduce_only:
            pos = self.positions.get(o.symbol)
            if not pos or pos.qty == 0 or (pos.qty > 0 and o.side == "BUY") or (pos.qty < 0 and o.side == "SELL"):
                o.status = "EXPIRED"
                return
            qty = min(qty, abs(pos.qty))
        fill = await self._execute(o.symbol, o.side, qty, px, maker=maker, client_id=o.client_id, order_id=o.order_id,
                                   reduce_only=o.reduce_only, ts=ts)
        o.filled_qty += qty
        o.avg_price = fill.price
        o.status = "FILLED"

    async def _execute(self, symbol: str, side: str, qty: float, px: float, maker: bool, client_id: str,
                       order_id: str, reduce_only: bool, kind: str | None = None, ts: int | None = None) -> Fill:
        ts = ts if ts is not None else self.now()
        pos = self.positions.setdefault(symbol, PaperPosition(leverage=self.leverages.get(symbol, 10)))
        signed = qty if side == "BUY" else -qty
        fee = qty * px * (self.maker_fee if maker else self.taker_fee)
        realized = 0.0
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            # open / add
            new_qty = pos.qty + signed
            pos.entry = (pos.entry * abs(pos.qty) + px * qty) / abs(new_qty) if new_qty else px
            pos.qty = new_qty
            k = kind or "ENTRY"
        else:
            closing = min(qty, abs(pos.qty))
            realized = (px - pos.entry) * closing * (1 if pos.qty > 0 else -1)
            remaining = abs(pos.qty) - closing
            if remaining <= 1e-12:
                pos.qty = 0.0
                pos.entry = 0.0
                if qty - closing > 1e-12:  # flipped
                    pos.qty = signed + (closing if signed < 0 else -closing)
                    pos.entry = px
            else:
                pos.qty = remaining if pos.qty > 0 else -remaining
            k = kind or "CLOSE"
        self.wallet += realized - fee
        self.realized += realized
        self.fees_paid += fee
        self.trade_count += 1
        fill = Fill(symbol=symbol, order_side=side, qty=qty, price=px, fee=fee, ts=ts, client_id=client_id,
                    order_id=order_id, reduce_only=reduce_only, maker=maker, kind=k)
        await self._emit_fill(fill)
        return fill

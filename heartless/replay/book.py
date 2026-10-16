"""Replay order books (F7): the conservative 1m-bar stand-in behind ctx.book when a replay runs with ORDERBOOK_STREAM.

A stored history has no depth data, so the replay cannot serve the depth streams. Instead of an empty book it installs
`ProxyBooks`, which answers the app's `books.features(symbol, now)` with exchange.orderbook.bar_book_proxy() over the
symbol's own live 1m frame (bars closed at or before `now`, nothing later): the same field names as the live book, with
a wider spread, a thinner book and no imbalance (docs/PERFORMANCE.md). The backtester can call the same function.
"""
from __future__ import annotations

from typing import Any

from heartless.exchange.orderbook import EMPTY_BOOK, BookFeatures, proxy_from_frame


class ProxyBooks:
    """Stands in for exchange.orderbook.OrderBooks inside a replayed app."""

    mode = "proxy"

    def __init__(self, app, depth_bps: float):
        self.app = app
        self.depth_bps = float(depth_bps)
        self.calls = 0

    def set_symbols(self, symbols: list[str]) -> None:  # the views define the symbols
        pass

    def on_reset(self, symbols: list[str] | None = None) -> None:
        pass

    async def on_message(self, ev: dict) -> None:
        pass

    async def close(self) -> None:
        pass

    def features(self, symbol: str, now: int | None = None) -> BookFeatures:
        view = self.app.views.get(symbol)
        if view is None or now is None:
            return EMPTY_BOOK
        self.calls += 1
        info = self.app.symbols.get(symbol)
        return proxy_from_frame(view.frames.get("1m"), int(now), self.depth_bps,
                                float(getattr(info, "tick_size", 0.0) or 0.0))

    def status(self) -> dict[str, Any]:
        return {"mode": "proxy", "symbols": len(self.app.views), "calls": self.calls}

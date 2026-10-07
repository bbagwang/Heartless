"""Tiny async event bus used to decouple engines from notifiers and the web UI."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)
Handler = Callable[[Any], Awaitable[None] | None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self.history: list[tuple[int, str, Any]] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, payload: Any = None) -> None:
        from heartless.util.timeutil import now_ms

        self.history.append((now_ms(), topic, payload))
        if len(self.history) > 2000:
            self.history = self.history[-1000:]
        for h in list(self._handlers.get(topic, [])) + list(self._handlers.get("*", [])):
            try:
                res = h(payload) if topic != "*" else h(payload)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001
                log.exception("event handler failed for %s", topic)

    def emit(self, topic: str, payload: Any = None) -> None:
        """Fire-and-forget from sync code inside the running loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.publish(topic, payload))

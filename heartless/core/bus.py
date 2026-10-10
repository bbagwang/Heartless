"""Tiny async event bus used to decouple engines from notifiers and the web UI."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)
Handler = Callable[[Any], Awaitable[None] | None]

HISTORY_MAX = 2000  # entries kept (trimmed to half when exceeded)
HISTORY_TEXT_CHARS = 300  # longest string kept per field in a history entry


def slim(payload: Any) -> Any:
    """What the history keeps of an event payload: scalars and short strings only. Payloads routinely carry big
    objects (a research cycle's full result, the daily report's status and equity curves, positions); keeping
    references to them in a 2000-entry history held multi-hundred-KB payloads alive for days (P5 soak). Dropped keys
    are listed under "_dropped" so the entry still says what the event carried."""
    if not isinstance(payload, dict):
        return payload if isinstance(payload, (str, int, float, bool, type(None))) else type(payload).__name__
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in payload.items():
        if isinstance(v, str):
            out[k] = v if len(v) <= HISTORY_TEXT_CHARS else v[:HISTORY_TEXT_CHARS] + "…"
        elif isinstance(v, (int, float, bool, type(None))):
            out[k] = v
        else:
            dropped.append(str(k))
    if dropped:
        out["_dropped"] = dropped
    return out


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self.history: list[tuple[int, str, Any]] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, payload: Any = None) -> None:
        from heartless.util.timeutil import now_ms

        self.history.append((now_ms(), topic, slim(payload)))
        if len(self.history) > HISTORY_MAX:
            self.history = self.history[-HISTORY_MAX // 2:]
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

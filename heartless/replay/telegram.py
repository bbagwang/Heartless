"""Telegram capture stub for replays.

The real `TelegramBot` runs (event subscriptions, formatting, command handling, the sender queue with its pacing);
only its HTTP client is swapped for an `httpx.MockTransport` that

* records every outgoing message with its virtual timestamp and the event/command that produced it, and
* answers `getUpdates` long polls in virtual time with scripted owner commands (`/status`, `/pause`, `/close BTCUSDT`,
  ...), delivered exactly at their scheduled instant. `@confirm` / `@cancel` press the button of the newest
  confirmation keyboard (`/close all`, `/kill`); `/close *` names an open position of the primary engine at delivery.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Callable

import httpx

from heartless.notify.telegram import TelegramBot
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, now_ms

OWNER_ID = 1


@dataclass
class ScriptedCommand:
    at_ms: int
    text: str


@dataclass
class SentMessage:
    ts: int
    topic: str  # bus topic, "cmd:/status", "startup", "transport:<method>" ...
    text: str
    keyboard: list | None = None


@dataclass
class TelegramCapture:
    """Mock Bot API: message log + scripted owner commands."""
    commands: list[ScriptedCommand] = field(default_factory=list)
    messages: list[SentMessage] = field(default_factory=list)
    api_calls: dict[str, int] = field(default_factory=dict)
    delivered: list[tuple[int, str]] = field(default_factory=list)  # (virtual ms, resolved text)
    listeners: list[Callable[[SentMessage], None]] = field(default_factory=list)
    resolve_symbol: Callable[[], str] | None = None  # for `/close *`
    _next_cmd: int = 0
    _update_id: int = 0

    def __post_init__(self) -> None:
        self.commands = sorted(self.commands, key=lambda c: c.at_ms)

    def record(self, topic: str, text: str, keyboard: list | None) -> None:
        msg = SentMessage(now_ms(), topic, text, keyboard)
        self.messages.append(msg)
        for fn in list(self.listeners):
            fn(msg)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.messages:
            out[m.topic] = out.get(m.topic, 0) + 1
        return dict(sorted(out.items()))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), base_url="https://api.telegram.org")

    # --- mock transport ------------------------------------------------------------------------------
    async def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        self.api_calls[method] = self.api_calls.get(method, 0) + 1
        try:
            params = json.loads(request.content or b"{}")
        except ValueError:
            params = {}
        if method == "getUpdates":
            return httpx.Response(200, json={"ok": True, "result": await self._updates(params)})
        if method == "sendMessage":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": self.api_calls[method],
                                                                    "chat": {"id": params.get("chat_id")}}})
        return httpx.Response(200, json={"ok": True, "result": True})

    async def _updates(self, params: dict) -> list[dict]:
        """Long poll: return the scripted commands that are due, else wait (virtually) for the next one or the poll
        timeout, whichever comes first."""
        due = self._due()
        if not due:
            wait_ms = int(params.get("timeout", 50) or 0) * 1000
            if self._next_cmd < len(self.commands):
                wait_ms = min(wait_ms, max(self.commands[self._next_cmd].at_ms - now_ms(), 0))
            loop = asyncio.get_running_loop()
            if hasattr(loop, "sleep_until_ms"):
                await loop.sleep_until_ms(now_ms() + wait_ms)
            else:
                await asyncio.sleep(wait_ms / 1000)
            due = self._due()
        return [self._update(c) for c in due]

    def _due(self) -> list[ScriptedCommand]:
        out = []
        now = now_ms()
        while self._next_cmd < len(self.commands) and self.commands[self._next_cmd].at_ms <= now:
            out.append(self.commands[self._next_cmd])
            self._next_cmd += 1
        return out

    def _update(self, cmd: ScriptedCommand) -> dict:
        self._update_id += 1
        text = cmd.text.strip()
        if text.lower().startswith("/close *"):
            text = "/close " + (self.resolve_symbol() if self.resolve_symbol else "BTCUSDT")
        self.delivered.append((now_ms(), text))
        user = {"id": OWNER_ID, "is_bot": False, "first_name": "owner"}
        if text in ("@confirm", "@cancel"):
            data = "cancel"
            if text == "@confirm":
                for m in reversed(self.messages):
                    cbs = [b.get("callback_data", "") for row in (m.keyboard or []) for b in row]
                    hit = next((c for c in cbs if c.startswith("cf:")), None)
                    if hit:
                        data = hit
                        break
            return {"update_id": self._update_id,
                    "callback_query": {"id": str(self._update_id), "from": user, "data": data,
                                       "message": {"message_id": 0, "chat": {"id": OWNER_ID, "type": "private"}}}}
        return {"update_id": self._update_id,
                "message": {"message_id": self._update_id, "from": user, "date": now_ms() // 1000,
                            "chat": {"id": OWNER_ID, "type": "private"}, "text": text}}


class CaptureTelegramBot(TelegramBot):
    """The production bot with a capture transport; remembers which event/command produced each message."""

    def __init__(self, app, capture: TelegramCapture):
        self.capture = capture
        self._ctx = "startup"
        super().__init__(app, client=capture.client())

    def send(self, text: str, keyboard: list[list[dict]] | None = None, chat_id: int | None = None) -> None:
        self.capture.record(self._ctx, text, keyboard)
        super().send(text, keyboard, chat_id)

    async def _on_event(self, topic: str, p: dict) -> None:
        prev, self._ctx = self._ctx, topic
        try:
            await super()._on_event(topic, p)
        finally:
            self._ctx = prev

    async def _handle_command(self, text: str, chat_id: int) -> None:
        prev, self._ctx = self._ctx, "cmd:" + text.split()[0].lower().split("@")[0]
        try:
            await super()._handle_command(text, chat_id)
        finally:
            self._ctx = prev

    async def _handle_callback(self, cq: dict) -> None:
        prev, self._ctx = self._ctx, "callback:" + str(cq.get("data", "")).split(":")[0]
        try:
            await super()._handle_callback(cq)
        finally:
            self._ctx = prev


_OFFSET = re.compile(r"^\+?(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?$")


def parse_command(spec: str, start_ms: int) -> ScriptedCommand:
    """`WHEN=TEXT`, WHEN an ISO date-time (UTC unless an offset is given) or an offset from the replay start like
    `+2d`, `+1d6h`, `+90m`. Example: `+2d=/pause`, `2026-07-03T06:00=/resume`."""
    when, sep, text = spec.partition("=")
    if not sep or not text.strip():
        raise ValueError(f"command {spec!r}: expected WHEN=TEXT")
    when = when.strip()
    m = _OFFSET.match(when)
    if when.startswith("+") and m and any(m.groups()):
        d, h, mi = (int(x or 0) for x in m.groups())
        at = start_ms + d * MS_DAY + h * MS_HOUR + mi * MS_MINUTE
    else:
        from heartless.learning.lab import parse_when

        at = parse_when(when)
    return ScriptedCommand(at, text.strip())

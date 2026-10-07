"""Regression tests for heartless/notify/telegram.py (fix round 17).

1. Pairing: fresh high-entropy code per start (not persisted), private-chat only, expiry, lockout after
   repeated wrong codes, TELEGRAM_OWNER_ID skips pairing.
2. _chunks: never an empty chunk, never a chunk longer than the limit (single over-long line).
3. api(): a 429 flood-control reply is retried (bounded) instead of dropping the request.
4. Pairing confirmation (which carries the web token) is never sent to the chat the /start came from.
"""
import asyncio
import tempfile
from pathlib import Path

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.store import Store
from heartless.notify import telegram as T
from heartless.notify.telegram import TelegramBot, _chunks

WEB_TOKEN = "SUPER-SECRET-WEB-TOKEN"


class _App:
    def __init__(self, store, owner=None):
        self.store = store
        kw = {"TELEGRAM_OWNER_ID": owner} if owner else {}
        self.s = Settings(_env_file=None, TELEGRAM_BOT_TOKEN="123:abc", **kw)
        self.bus = EventBus()
        self.mode = "paper"
        self.web_token = WEB_TOKEN

    def startup_summary(self):
        return f"started http://<host>:8080/?token={self.web_token}"


def _store():
    return Store(Path(tempfile.mkdtemp()) / "t.db")


def _bot(store=None, owner=None, stub_send=True):
    store = store or _store()
    bot = TelegramBot(_App(store, owner))
    sent: list[tuple[str, int | None]] = []
    if stub_send:  # the end-to-end _sender tests keep the real send() -> queue path
        bot.send = lambda text, keyboard=None, chat_id=None: sent.append((text, chat_id))
    return bot, store, sent


def _msg(uid, text, chat_id=None, chat_type=None):
    chat = {"id": uid if chat_id is None else chat_id}
    if chat_type:
        chat["type"] = chat_type
    return {"message": {"from": {"id": uid}, "chat": chat, "text": text}}


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _rate_limited(retry_after=0.0):
    return {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 1",
            "parameters": {"retry_after": retry_after}}


# --- finding 1: pairing code hardening ------------------------------------------------------------

def test_pairing_code_is_fresh_high_entropy_and_not_persisted():
    store = _store()
    store.set("telegram.pairing", "123456")  # stale code left behind by an older version
    bot, _, sent = _bot(store)
    assert bot.owner_id is None and bot.pairing_code
    assert len(bot.pairing_code) >= 16 and not bot.pairing_code.isdigit()
    # not reused from the store and not written back to it
    assert bot.pairing_code != "123456"
    assert store.get("telegram.pairing") != bot.pairing_code
    # every process start gets its own code
    other, _, _ = _bot()
    assert other.pairing_code != bot.pairing_code

    async def run():
        await bot._handle_update(_msg(1, "/start 123456"))  # the stale persisted code must not work
        assert bot.owner_id is None and not sent

    asyncio.run(run())


def test_pairing_locks_out_after_repeated_wrong_codes():
    bot, store, sent = _bot()
    code = bot.pairing_code

    async def run():
        # bare /start, chatter and non-/start text from strangers are not counted as attempts
        await bot._handle_update(_msg(5, "/start"))
        await bot._handle_update(_msg(5, "hello"))
        await bot._handle_update(_msg(5, "/status"))
        assert bot._pairing_failures == 0 and bot.pairing_code == code
        # a non-ASCII guess must not crash the handler and counts as a failure
        await bot._handle_update(_msg(6, "/start 한글코드"))
        assert bot._pairing_failures == 1 and bot.pairing_code == code
        # attempts spread across many accounts share one global counter
        for i in range(T.PAIRING_MAX_FAILURES - 1):
            assert bot.pairing_code == code
            await bot._handle_update(_msg(100 + i, f"/start {100000 + i}"))
        assert bot._pairing_failures == T.PAIRING_MAX_FAILURES
        assert bot.pairing_code is None  # disarmed
        # even the correct code no longer binds until a restart
        await bot._handle_update(_msg(42, f"/start {code}"))
        assert bot.owner_id is None and store.get("telegram.owner") is None and not sent

    asyncio.run(run())


def test_pairing_code_expires():
    bot, store, sent = _bot()
    code = bot.pairing_code
    bot._pairing_born -= T.PAIRING_TTL_SEC + 1

    async def run():
        await bot._handle_update(_msg(42, f"/start {code}"))
        assert bot.owner_id is None and bot.pairing_code is None and not sent
        assert store.get("telegram.owner") is None

    asyncio.run(run())


def test_pairing_still_works_within_limits_and_clears_store_key():
    store = _store()
    store.set("telegram.pairing", "123456")
    bot, _, sent = _bot(store)
    code = bot.pairing_code

    async def run():
        for i in range(T.PAIRING_MAX_FAILURES - 1):
            await bot._handle_update(_msg(1, f"/start wrong{i}"))
        assert bot.pairing_code == code
        await bot._handle_update(_msg(42, f"/start {code}", chat_type="private"))
        assert bot.owner_id == 42 and store.get("telegram.owner") == 42
        assert bot.pairing_code is None and store.get("telegram.pairing") is None
        assert bot._is_owner(42) and not bot._is_owner(1)

    asyncio.run(run())


def test_preset_owner_id_skips_pairing_entirely():
    bot, _, _ = _bot(owner=99)
    assert bot.owner_id == 99 and bot.pairing_code is None

    async def run():
        await bot._handle_update(_msg(42, "/start anything"))
        assert bot.owner_id == 99

    asyncio.run(run())


# --- finding 4: confirmation (web token) never goes to a group chat ----------------------------------

def test_pairing_from_group_is_rejected_and_token_never_sent_to_group():
    bot, store, sent = _bot()
    code = bot.pairing_code
    group = -100123

    async def run():
        # supergroup with the correct code: no pairing, nothing sent anywhere
        await bot._handle_update(_msg(42, f"/start {code}", chat_id=group, chat_type="supergroup"))
        assert bot.owner_id is None and not sent
        # chat without a type but whose id is not the sender's: also not a private chat
        await bot._handle_update(_msg(42, f"/start {code}", chat_id=group))
        assert bot.owner_id is None and not sent
        # rejected group attempts with the right code are not counted as failed guesses
        assert bot._pairing_failures == 0 and bot.pairing_code == code
        # pairing from the private chat works and every message is routed to the owner, not the group
        await bot._handle_update(_msg(42, f"/start {code}", chat_type="private"))
        assert bot.owner_id == 42
        assert sent and all(cid in (None, 42) for _, cid in sent)
        assert any(WEB_TOKEN in text for text, _ in sent)
        assert not any(cid == group for _, cid in sent)

    asyncio.run(run())


def test_sender_routes_pairing_confirmation_to_owner_private_chat():
    """End to end through the real send()/_sender(): the sendMessage chat_id is the owner's id."""
    bot, _, _ = _bot(stub_send=False)
    code = bot.pairing_code
    calls = []

    async def post(url, json=None):
        calls.append((url.rsplit("/", 1)[-1], json))
        return _Resp({"ok": True, "result": {}})

    bot.client.post = post

    async def run():
        task = asyncio.create_task(bot._sender())
        await bot._handle_update(_msg(42, f"/start {code}", chat_type="private"))
        for _ in range(200):
            if len([c for c in calls if c[0] == "sendMessage"]) >= 2:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        msgs = [j for m, j in calls if m == "sendMessage"]
        assert len(msgs) >= 2
        assert all(j["chat_id"] == 42 for j in msgs)
        assert any(WEB_TOKEN in j["text"] for j in msgs)

    asyncio.run(run())


# --- finding 2: _chunks ------------------------------------------------------------------------------

def test_chunks_single_long_line_has_no_empty_or_oversized_chunk():
    out = _chunks("x" * 5000, 3900)
    assert out == ["x" * 3900, "x" * 1100]
    assert all(0 < len(c) <= 3900 for c in out)


def test_chunks_long_first_line_followed_by_text():
    text = "y" * 4000 + "\nshort"
    out = _chunks(text, 3900)
    assert all(0 < len(c) <= 3900 for c in out)
    assert out[0] == "y" * 3900
    assert "".join(c.replace("\n", "") for c in out) == text.replace("\n", "")


def test_chunks_mixed_lines_preserve_content_and_short_lines():
    lines = ["<b>AI 리뷰</b>", "p" * 8000, "", "tail line", "q" * 3900, "end"]
    text = "\n".join(lines)
    out = _chunks(text, 3900)
    assert all(0 < len(c) <= 3900 for c in out)
    assert "".join(c.replace("\n", "") for c in out) == text.replace("\n", "")
    joined = "\n".join(out)
    for short in ("<b>AI 리뷰</b>", "tail line", "end"):
        assert short in joined.split("\n")


def test_chunks_short_text_and_exact_limit_unchanged():
    assert _chunks("hello\nworld", 3900) == ["hello\nworld"]
    assert _chunks("z" * 3900, 3900) == ["z" * 3900]
    multi = "\n".join(["a" * 10] * 50)  # 549 chars, lines kept whole
    out = _chunks(multi, 100)
    assert all(0 < len(c) <= 100 for c in out) and "\n".join(out) == multi


# --- finding 3: 429 retry ----------------------------------------------------------------------------

def test_api_retries_identical_request_after_429_and_succeeds():
    bot, _, _ = _bot()
    bodies = [_rate_limited(0.01), {"ok": True, "result": {"message_id": 1}}]
    calls = []

    async def post(url, json=None):
        calls.append((url, json))
        return _Resp(bodies.pop(0))

    bot.client.post = post

    async def run():
        res = await bot.api("sendMessage", chat_id=42, text="hi", parse_mode="HTML")
        assert res.get("ok") is True
        assert len(calls) == 2 and calls[0] == calls[1]
        assert calls[0][1] == {"chat_id": 42, "text": "hi", "parse_mode": "HTML"}

    asyncio.run(run())


def test_api_gives_up_after_bounded_attempts_and_does_not_retry_other_errors():
    bot, _, _ = _bot()
    calls = []

    async def post_429(url, json=None):
        calls.append(json)
        return _Resp(_rate_limited(0.0))

    bot.client.post = post_429

    async def run():
        res = await bot.api("getUpdates", offset=7)
        assert res.get("ok") is False and res.get("error_code") == 429
        assert len(calls) == T.TELEGRAM_API_ATTEMPTS
        # a non-429 failure is returned immediately (no retry storm on e.g. parse errors)
        calls.clear()

        async def post_400(url, json=None):
            calls.append(json)
            return _Resp({"ok": False, "error_code": 400, "description": "Bad Request: can't parse entities"})

        bot.client.post = post_400
        res = await bot.api("sendMessage", chat_id=1, text="x")
        assert res.get("error_code") == 400 and len(calls) == 1

    asyncio.run(run())


def test_sender_delivers_notification_despite_flood_control():
    """A burst of closes: the rate-limited sendMessage is re-sent, so the owner still gets the message."""
    bot, _, _ = _bot(owner=42, stub_send=False)
    bodies = [_rate_limited(0.01), {"ok": True, "result": {}}]
    calls = []

    async def post(url, json=None):
        calls.append((url.rsplit("/", 1)[-1], json))
        return _Resp(bodies.pop(0) if bodies else {"ok": True, "result": {}})

    bot.client.post = post

    async def run():
        task = asyncio.create_task(bot._sender())
        bot.send("BTCUSDT closed -12.3 USDT")
        for _ in range(200):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        texts = [j["text"] for m, j in calls if m == "sendMessage"]
        assert texts == ["BTCUSDT closed -12.3 USDT", "BTCUSDT closed -12.3 USDT"]
        assert all(j["chat_id"] == 42 for _, j in calls)

    asyncio.run(run())

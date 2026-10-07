import asyncio
import tempfile
from pathlib import Path

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.store import Store
from heartless.notify.telegram import TelegramBot


class _App:
    def __init__(self, store):
        self.store = store
        self.s = Settings(_env_file=None, TELEGRAM_BOT_TOKEN="123:abc")
        self.bus = EventBus()
        self.mode = "paper"

    def startup_summary(self):
        return "started"


def test_pairing_binds_first_user_with_code_and_ignores_others():
    store = Store(Path(tempfile.mkdtemp()) / "t.db")
    bot = TelegramBot(_App(store))
    sent = []
    bot.send = lambda text, keyboard=None, chat_id=None: sent.append(text)
    code = bot.pairing_code
    assert code and len(code) >= 16  # high-entropy token_urlsafe(12), no longer a 6-digit number

    async def run():
        # stranger with wrong code
        await bot._handle_update({"message": {"from": {"id": 1}, "chat": {"id": 1}, "text": "/start 000000"}})
        assert bot.owner_id is None and not sent
        # correct code binds
        await bot._handle_update({"message": {"from": {"id": 42}, "chat": {"id": 42}, "text": f"/start {code}"}})
        assert bot.owner_id == 42 and store.get("telegram.owner") == 42
        # second user with the (now cleared) code is ignored
        await bot._handle_update({"message": {"from": {"id": 7}, "chat": {"id": 7}, "text": f"/start {code}"}})
        assert bot.owner_id == 42
        assert bot._is_owner(42) and not bot._is_owner(7)

    asyncio.run(run())

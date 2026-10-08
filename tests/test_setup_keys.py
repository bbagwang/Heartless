"""Runtime credential setup: secrets file, verified key registration, web endpoints and Telegram /setkeys."""
import asyncio
import os
import stat

import httpx

from heartless.config import Settings
from heartless.core.secrets import apply_to_settings, load_secrets, mask, save_secrets


def test_secrets_file_is_private_and_overlays_empty_settings(tmp_path):
    save_secrets(tmp_path, {"BINANCE_API_KEY": "k" * 20, "BINANCE_API_SECRET": "s" * 20, "BOGUS": "x"})
    mode = stat.S_IMODE(os.stat(tmp_path / "secrets.json").st_mode)
    assert mode == 0o600
    data = load_secrets(tmp_path)
    assert set(data) == {"BINANCE_API_KEY", "BINANCE_API_SECRET"}
    s = Settings(_env_file=None, BINANCE_API_KEY="")
    apply_to_settings(s, data)
    assert s.live_capable and mask(s.binance_api_key).startswith("kkkk")
    env = Settings(_env_file=None, BINANCE_API_KEY="fromenv", BINANCE_API_SECRET="envsecret")
    apply_to_settings(env, data)
    assert env.binance_api_key == "fromenv"  # environment wins
    save_secrets(tmp_path, {"BINANCE_API_KEY": ""})
    assert "BINANCE_API_KEY" not in load_secrets(tmp_path)


class _FakeRest:
    calls = []

    def __init__(self, key, secret, testnet):
        self.key = key

    async def sync_time(self):
        return None

    async def account(self):
        if self.key == "bad":
            raise RuntimeError("Invalid API-key")
        return {"canTrade": True, "totalWalletBalance": "1234.5"}

    async def close(self):
        return None


def _app(tmp_path, monkeypatch):
    import heartless.exchange.binance_rest as br
    from heartless.app import Heartless

    monkeypatch.setattr(br, "BinanceRest", _FakeRest)
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="")
    app = Heartless(s)
    swapped = []
    app.rest.set_credentials = lambda k, sec, tn=None: swapped.append((k, sec, tn))
    return app, swapped


def test_set_credentials_verifies_persists_and_keeps_paper(tmp_path, monkeypatch):
    app, swapped = _app(tmp_path, monkeypatch)

    async def run():
        bad = await app.set_credentials("bad", "x" * 20)
        assert bad.startswith("키 검증 실패") and not load_secrets(tmp_path)
        ok = await app.set_credentials("good" * 5, "y" * 20)
        assert ok.startswith("키 등록 완료") and app.mode == "paper"
        assert load_secrets(tmp_path)["BINANCE_API_KEY"] == "good" * 5 and swapped[-1][0] == "good" * 5
        assert app.s.live_capable

    asyncio.run(run())


def test_web_settings_endpoints_require_token_and_local_or_https(tmp_path, monkeypatch):
    from heartless.web.app import WebServer

    app, _ = _app(tmp_path, monkeypatch)
    web = WebServer(app)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.api, client=("127.0.0.1", 1234)),
                                     base_url="http://test") as c:
            assert (await c.get("/api/settings")).status_code == 401
            h = {"x-token": app.web_token}
            st = (await c.get("/api/settings", headers=h)).json()
            assert st["binance_api_key"] == "" and st["mode"] == "paper"
            r = (await c.post("/api/settings/keys", json={"api_key": "good" * 5, "api_secret": "z" * 20}, headers=h)).json()
            assert r["ok"] is True
            assert (await c.get("/api/settings", headers=h)).json()["binance_api_key"].startswith("good")
            r = await c.post("/api/settings/telegram", json={"token": "nope"}, headers=h)
            assert r.status_code == 400
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.api, client=("203.0.113.9", 1)),
                                     base_url="http://test") as remote:
            r = await remote.post("/api/settings/keys", json={"api_key": "a", "api_secret": "b"}, headers={"x-token": app.web_token})
            assert r.status_code == 400  # plain HTTP from a remote address: refused

    asyncio.run(run())


def test_telegram_setkeys_deletes_the_message_and_registers(tmp_path):
    from heartless.core.bus import EventBus
    from heartless.core.store import Store
    from heartless.notify.telegram import TelegramBot

    class _A:
        def __init__(self):
            self.store = Store(tmp_path / "t.db")
            self.s = Settings(_env_file=None, TELEGRAM_BOT_TOKEN="123:abc", TELEGRAM_OWNER_ID=42)
            self.bus = EventBus()
            self.mode = "paper"
            self.got = None

        def startup_summary(self):
            return ""

        async def set_credentials(self, k, s, tn):
            self.got = (k, s, tn)
            return "키 등록 완료"

    app = _A()
    bot = TelegramBot(app)
    calls, sent = [], []

    async def api(method, **params):
        calls.append((method, params))
        return {"ok": True}

    bot.api = api
    bot.send = lambda text, keyboard=None, chat_id=None: sent.append(text)
    asyncio.run(bot._handle_update({"message": {"message_id": 77, "from": {"id": 42}, "chat": {"id": 42, "type": "private"},
                                                "text": "/setkeys KEY123 SECRET456 testnet"}}))
    assert ("deleteMessage", {"chat_id": 42, "message_id": 77}) in calls
    assert app.got == ("KEY123", "SECRET456", True) and any("키 등록 완료" in t for t in sent)
    # strangers cannot use it (and their message is not even processed)
    app.got = None
    asyncio.run(bot._handle_update({"message": {"message_id": 78, "from": {"id": 7}, "chat": {"id": 7, "type": "private"},
                                                "text": "/setkeys A B"}}))
    assert app.got is None

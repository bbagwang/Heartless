"""Owner-provided credentials entered at runtime (web settings page or Telegram /setkeys).

They are stored in `<data_dir>/secrets.json` with 0600 permissions and overlaid on the environment at start-up,
so the only thing an owner ever has to provide is the Binance API key pair (and a Telegram bot token if they
want Telegram) - no file editing required. Environment variables / .env still take precedence when set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KEYS = ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "BINANCE_TESTNET", "ANTHROPIC_API_KEY")


def secrets_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "secrets.json"


def load_secrets(data_dir: str | Path) -> dict:
    p = secrets_path(data_dir)
    try:
        data = json.loads(p.read_text())
        return {k: v for k, v in data.items() if k in KEYS and isinstance(v, (str, bool))}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def save_secrets(data_dir: str | Path, updates: dict) -> dict:
    """Merge `updates` (empty values remove a key) into the secrets file atomically with owner-only permissions."""
    p = secrets_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_secrets(data_dir)
    for k, v in updates.items():
        if k not in KEYS:
            continue
        if v in (None, ""):
            data.pop(k, None)
        else:
            data[k] = v
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return data


def apply_to_settings(settings, data: dict) -> None:
    """Fill settings fields that the environment left empty."""
    if data.get("BINANCE_API_KEY") and not settings.binance_api_key:
        settings.binance_api_key = data["BINANCE_API_KEY"]
    if data.get("BINANCE_API_SECRET") and not settings.binance_api_secret:
        settings.binance_api_secret = data["BINANCE_API_SECRET"]
    if data.get("TELEGRAM_BOT_TOKEN") and not settings.telegram_bot_token:
        settings.telegram_bot_token = data["TELEGRAM_BOT_TOKEN"]
    if data.get("ANTHROPIC_API_KEY") and not settings.anthropic_api_key:
        settings.anthropic_api_key = data["ANTHROPIC_API_KEY"]
    if "BINANCE_TESTNET" in data and "BINANCE_TESTNET" not in os.environ:
        settings.binance_testnet = str(data["BINANCE_TESTNET"]).lower() in ("1", "true", "yes", "on")


def mask(value: str) -> str:
    if not value:
        return ""
    return value[:4] + "…" + value[-4:] if len(value) > 10 else "…"

from __future__ import annotations

import secrets
import time


def short_id(prefix: str = "") -> str:
    return f"{prefix}{int(time.time() * 1000) % 10_000_000_000:010d}{secrets.token_hex(2)}"


def client_order_id(tag: str) -> str:
    """Binance allows up to 36 chars for newClientOrderId / clientAlgoId."""
    base = f"HL{tag}{secrets.token_hex(6)}"
    return base[:36]


def token_hex(n: int = 24) -> str:
    return secrets.token_hex(n)

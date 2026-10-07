"""Dynamic trading universe: liquid USDT perpetuals ranked by 24h quote volume."""
from __future__ import annotations

import logging

from heartless.core.models import SymbolInfo
from heartless.util.timeutil import MS_DAY, now_ms

log = logging.getLogger(__name__)

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "USD1", "USDE", "XUSD", "AEUR", "BFUSD", "PAXG", "XAUT"}


def _ineligible_reason(info: SymbolInfo) -> str | None:
    """Instrument-level check shared by candidates and ALWAYS_INCLUDE: USDT-margined, perpetual, trading."""
    if info.quote != "USDT":
        return f"quote={info.quote}"
    if info.contract_type != "PERPETUAL":
        return f"contractType={info.contract_type}"
    if info.status != "TRADING":
        return f"status={info.status}"
    return None


def select_universe(infos: dict[str, SymbolInfo], tickers24h: list[dict], size: int, min_quote_volume: float,
                    always: list[str], min_age_days: int = 14) -> list[str]:
    now = now_ms()
    vol: dict[str, float] = {}
    for t in tickers24h:
        try:
            vol[t["symbol"]] = float(t.get("quoteVolume", 0.0))
        except Exception:
            continue
    candidates = []
    for sym, info in infos.items():
        if _ineligible_reason(info) is not None:
            continue
        if info.base in STABLE_BASES:
            continue
        if info.onboard_date and now - info.onboard_date < min_age_days * MS_DAY and sym not in always:
            continue
        qv = vol.get(sym, 0.0)
        if qv < min_quote_volume and sym not in always:
            continue
        candidates.append((qv, sym))
    candidates.sort(reverse=True)
    chosen: list[str] = []
    for s in always:
        info = infos.get(s)
        reason = "not in exchangeInfo" if info is None else _ineligible_reason(info)
        if reason is not None:
            log.warning("always-include %s skipped (%s)", s, reason)
            continue
        if s not in chosen:
            chosen.append(s)
    for _, sym in candidates:
        if len(chosen) >= size:
            break
        if sym not in chosen:
            chosen.append(sym)
    log.info("universe selected (%d): %s", len(chosen), ",".join(chosen))
    return chosen

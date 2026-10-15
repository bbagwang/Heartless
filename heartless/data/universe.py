"""Which symbols the bot trades (live) and which symbols research evaluates (historical), by rule instead of by name.

Live (select_live_universe): every USDT-margined perpetual that is trading, at least MIN_AGE_DAYS listed, not a
stablecoin, with 24h quote volume >= UNIVERSE_MIN_QUOTE_VOLUME, a tick no coarser than MAX_TICK_BPS of the price and,
when book tickers are supplied, a spread of at most MAX_SPREAD_BPS; the most liquid `size` of them trade. ALWAYS_INCLUDE
is a data subscription (BTC is the regime and market-context reference every alpha reads): those symbols are
subscribed even when they fail a liquidity rule, and they trade only when they pass the same rules as every other
symbol. select_universe() keeps the older list interface for app.py (see its docstring).

Research (research_universe): the survivorship-free counterpart. For each calendar month m, the USDT perpetuals of
the archive inventory (heartless/data/symbols_inventory.py, delisted ones included) are ranked by the median daily
quote volume of month m-1, so the selection at m uses only data that existed before m. A window's research universe is
every symbol that was top-N in at least one of its months, including the ones that were delisted later, so backtests
meet the delistings instead of only the coins that survived until today.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from heartless.core.models import SymbolInfo
from heartless.util.timeutil import MS_DAY, now_ms

log = logging.getLogger(__name__)

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "USD1", "USDE", "XUSD", "AEUR", "BFUSD", "PAXG", "XAUT"}
# live eligibility defaults (Settings: UNIVERSE_MIN_AGE_DAYS, UNIVERSE_MAX_SPREAD_BPS, UNIVERSE_MAX_TICK_BPS)
MIN_AGE_DAYS = 30  # a new listing's first weeks are price discovery with thin books and no indicator history
MAX_SPREAD_BPS = 5.0  # a quoted spread above this costs more than the strategies' whole per-trade cost budget
MAX_TICK_BPS = 10.0  # one tick above 0.1% of the price makes stops, targets and limit prices coarse


def _ineligible_reason(info: SymbolInfo) -> str | None:
    """Instrument-level check shared by candidates and ALWAYS_INCLUDE: USDT-margined, perpetual, trading."""
    if info.quote != "USDT":
        return f"quote={info.quote}"
    if info.contract_type != "PERPETUAL":
        return f"contractType={info.contract_type}"
    if info.status != "TRADING":
        return f"status={info.status}"
    return None


def _float(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


@dataclass
class UniverseSelection:
    trade: list[str]  # symbols that pass every rule, most liquid first, at most `size`
    reference: list[str]  # ALWAYS_INCLUDE symbols that exist and pass the instrument checks (data subscription)
    rejected: dict[str, str] = field(default_factory=dict)  # USDT perpetual -> first rule it failed

    @property
    def subscribed(self) -> list[str]:
        """Every symbol whose market data is needed: the reference symbols first, then the traded ones."""
        return self.reference + [s for s in self.trade if s not in self.reference]


def eligibility(info: SymbolInfo, quote_volume: float, now: int, min_quote_volume: float,
                min_age_days: float = MIN_AGE_DAYS, last_price: float = 0.0, max_tick_bps: float | None = MAX_TICK_BPS,
                spread_bps: float | None = None, max_spread_bps: float | None = MAX_SPREAD_BPS) -> str | None:
    """The first live rule a symbol fails, or None when it may trade. Only facts of the symbol itself are used (no
    name list): instrument filters, stablecoin base, listing age, 24h quote volume, tick size against the last price and,
    when known, the quoted spread."""
    why = _ineligible_reason(info)
    if why is not None:
        return why
    if info.base in STABLE_BASES:
        return "stablecoin"
    if info.onboard_date and now - info.onboard_date < min_age_days * MS_DAY:
        return f"listed {(now - info.onboard_date) / MS_DAY:.1f}d ago < {min_age_days}d"
    if quote_volume < min_quote_volume:
        return f"24h quote volume {quote_volume:,.0f} < {min_quote_volume:,.0f}"
    if max_tick_bps is not None and last_price > 0 and info.tick_size / last_price * 1e4 > max_tick_bps:
        return f"tick {info.tick_size / last_price * 1e4:.1f}bp > {max_tick_bps}bp"
    if max_spread_bps is not None and spread_bps is not None and spread_bps > max_spread_bps:
        return f"spread {spread_bps:.1f}bp > {max_spread_bps}bp"
    return None


def spreads_bps(book_tickers: list[dict] | None) -> dict[str, float]:
    """Quoted spread in bp per symbol from /fapi/v1/ticker/bookTicker rows (symbols without a two-sided book omitted)."""
    out: dict[str, float] = {}
    for b in book_tickers or []:
        bid, ask = _float(b.get("bidPrice")), _float(b.get("askPrice"))
        if bid > 0 and ask >= bid:
            out[str(b.get("symbol"))] = (ask - bid) / ((ask + bid) / 2) * 1e4
    return out


def select_live_universe(infos: dict[str, SymbolInfo], tickers24h: list[dict], size: int, min_quote_volume: float,
                         reference: list[str] | tuple[str, ...] = (), min_age_days: float = MIN_AGE_DAYS,
                         book_tickers: list[dict] | None = None, max_spread_bps: float | None = MAX_SPREAD_BPS,
                         max_tick_bps: float | None = MAX_TICK_BPS, now: int | None = None) -> UniverseSelection:
    """The live trading universe by rule (module docstring). `reference` symbols are never a trading guarantee."""
    now = now_ms() if now is None else int(now)
    vol: dict[str, float] = {}
    last: dict[str, float] = {}
    for t in tickers24h:
        sym = t.get("symbol")
        if sym is None:
            continue
        vol[sym] = _float(t.get("quoteVolume", 0.0))
        last[sym] = _float(t.get("lastPrice", 0.0))
    spreads = spreads_bps(book_tickers)
    ranked: list[tuple[float, str]] = []
    rejected: dict[str, str] = {}
    for sym, info in infos.items():
        if info.quote != "USDT" or info.contract_type != "PERPETUAL":
            continue
        why = eligibility(info, vol.get(sym, 0.0), now, min_quote_volume, min_age_days, last.get(sym, 0.0),
                          max_tick_bps, spreads.get(sym), max_spread_bps)
        if why is None:
            ranked.append((-vol.get(sym, 0.0), sym))
        else:
            rejected[sym] = why
    ranked.sort()
    ref: list[str] = []
    for s in reference:
        info = infos.get(s)
        why = "not in exchangeInfo" if info is None else _ineligible_reason(info)
        if why is not None:
            log.warning("always-include %s skipped (%s)", s, why)
            continue
        if s not in ref:
            ref.append(s)
    return UniverseSelection(trade=[s for _, s in ranked[:max(0, int(size))]], reference=ref, rejected=rejected)


def select_universe(infos: dict[str, SymbolInfo], tickers24h: list[dict], size: int, min_quote_volume: float,
                    always: list[str], min_age_days: float = MIN_AGE_DAYS, **rules) -> list[str]:
    """List interface used by app.py: the subscribed symbols of select_live_universe, `always` (ALWAYS_INCLUDE) first,
    capped at `size` (or the number of `always` symbols when larger).

    The app trades every symbol of this list, so until it subscribes `UniverseSelection.reference` as data only
    (context-only streams, BTC regime from that stream), a reference symbol that fails a liquidity rule is still
    traded; with the default ALWAYS_INCLUDE=BTCUSDT that never happens in practice (BTC is the most liquid perpetual).
    `rules`: book_tickers, max_spread_bps, max_tick_bps, now (select_live_universe)."""
    sel = select_live_universe(infos, tickers24h, size, min_quote_volume, always, min_age_days, **rules)
    chosen = sel.subscribed[:max(int(size), len(sel.reference))]
    log.info("universe selected (%d): %s", len(chosen), ",".join(chosen))
    return chosen


def live_selection(settings, infos: dict[str, SymbolInfo], tickers24h: list[dict],
                   book_tickers: list[dict] | None = None) -> UniverseSelection:
    """select_live_universe with every rule taken from the settings (the call app.py should make)."""
    return select_live_universe(infos, tickers24h, settings.universe_size, settings.universe_min_quote_volume,
                                settings.always_include_list, settings.universe_min_age_days, book_tickers,
                                settings.universe_max_spread_bps, settings.universe_max_tick_bps)


# --- research universe (survivorship-free, from the archive inventory) ---------------------------------------------------
RESEARCH_MIN_AGE_DAYS = MIN_AGE_DAYS  # the live listing-age rule, applied at the start of the ranked month


def _month_ms(month: str) -> int:
    from heartless.data.symbols_inventory import _month_ms as mm

    return mm(month)


def prev_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - (m == 1):04d}-{12 if m == 1 else m - 1:02d}"


def months_of(start: int, end: int) -> list[str]:
    """Calendar months ("YYYY-MM") overlapping [start, end] (ms)."""
    from heartless.data.symbols_inventory import _months_between, month_str

    return _months_between(month_str(start), month_str(end)) if end >= start else []


def listed_at(row: dict) -> int | None:
    """Listing time of an inventory row: its first hour with trades when the archive's first month is in the stored
    blocks, else the start of its first archive month (the archive does not go further back, so it is an upper bound
    of the age only for the few symbols listed during that month)."""
    first_month = row.get("first_month")
    if not first_month:
        return None
    m0 = _month_ms(first_month)
    ft = row.get("first_trade")
    return int(ft) if ft is not None and ft < m0 + 31 * MS_DAY else m0


def eligible_research(row: dict) -> str | None:
    """Instrument rules of the research universe (the live ones that the archive can tell): a USDT perpetual (not a
    delivery contract or an archive copy of a settled one) whose asset is not a stablecoin."""
    if row.get("kind") != "perpetual":
        return f"kind={row.get('kind')}"
    if row.get("quote") != "USDT":
        return f"quote={row.get('quote')}"
    if row.get("asset") in STABLE_BASES or row.get("base") in STABLE_BASES:
        return "stablecoin"
    return None


def month_ranking(store, month: str, inventory: list[dict] | None = None,
                  min_age_days: float = RESEARCH_MIN_AGE_DAYS) -> list[dict]:
    """Every eligible symbol ranked for calendar month `month` by the median daily quote volume of the previous month
    (days without bars count as 0; heartless/data/symbols_inventory.encode_month), most liquid first, ties by name.
    Only data before the month is used: the previous month's volumes and the listing time. Symbols listed less than
    `min_age_days` before the month starts, or without volume in the previous month, are not ranked."""
    from heartless.data.symbols_inventory import FUT_TABLE

    inv = {r["symbol"]: r for r in (inventory if inventory is not None else store.load_symbol_inventory())}
    m0 = _month_ms(month)
    ranked = []
    for b in store.kline_block_index(FUT_TABLE, month=_month_ms(prev_month(month))):
        row = inv.get(b["symbol"])
        if row is None or eligible_research(row) is not None or not b["n"] or not (b["med_dqv"] or 0) > 0:
            continue
        listed = listed_at(row)
        if listed is None or m0 - listed < min_age_days * MS_DAY:
            continue
        ranked.append({"symbol": b["symbol"], "med_dqv": float(b["med_dqv"]), "days_traded": b["days_traded"],
                       "listed": listed, "last_trade": row.get("last_trade"), "spot_symbol": row.get("spot_symbol")})
    ranked.sort(key=lambda r: (-r["med_dqv"], r["symbol"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def research_universe(store, start: int, end: int, top: int = 40) -> dict:
    """The research universe of [start, end]: per month the top-`top` symbols of month_ranking, and every symbol that
    was among them in at least one month of the window (delisted ones included), with the months it was a member and
    its last hour with trades. Months the stored 1h data cannot rank (the previous month is not built) are listed in
    "unranked" instead of being silently dropped."""
    inv = store.load_symbol_inventory()
    per_month: dict[str, list[str]] = {}
    unranked: list[str] = []
    members: dict[str, list[str]] = {}
    for mo in months_of(start, end):
        ranked = month_ranking(store, mo, inv)
        if not ranked:
            unranked.append(mo)
            continue
        per_month[mo] = [r["symbol"] for r in ranked[:top]]
        for s in per_month[mo]:
            members.setdefault(s, []).append(mo)
    inv_by = {r["symbol"]: r for r in inv}
    symbols = sorted(members, key=lambda s: (-len(members[s]), s))
    return {"top": top, "months": per_month, "unranked": unranked, "symbols": symbols, "members": members,
            "last_trade": {s: inv_by.get(s, {}).get("last_trade") for s in symbols}}


def parse_top(name: str | None) -> int | None:
    """N of a "topN" universe name (top40 -> 40), None for other names."""
    if not name or not name.lower().startswith("top"):
        return None
    tail = name[3:]
    if not tail.isdigit() or int(tail) <= 0:
        raise ValueError(f"universe {name!r}: expected topN with N > 0 (e.g. top40)")
    return int(tail)


def stored_candle_symbols(candle_db, symbols: list[str]) -> set[str]:
    """Which of `symbols` have 1m candles in the candle database, read through a read-only connection (one indexed
    lookup per symbol; the long-history database is never opened for writing)."""
    import sqlite3
    from pathlib import Path

    if not Path(candle_db).exists():
        return set()
    con = sqlite3.connect(f"file:{Path(candle_db).resolve()}?mode=ro", uri=True)
    try:
        have = set()
        for s in symbols:
            if con.execute("SELECT 1 FROM candles WHERE symbol=? LIMIT 1", (s,)).fetchone():
                have.add(s)
        return have
    except sqlite3.OperationalError:  # no candles table
        return set()
    finally:
        con.close()


def resolve_for_run(universe_db, candle_db, start: int, end: int, top: int) -> dict:
    """The topN research universe of [start, end] for a lab / discovery run over `candle_db`: "research" = every
    member (research_universe), "symbols" = those with stored 1m candles (what the run evaluates), "skipped" = the
    members without 1m candles, to be reported (fetch them with `heartless fetch --start ... --symbols ...`).
    ValueError when the universe database is missing or a month of the window cannot be ranked yet."""
    from pathlib import Path

    from heartless.core.store import Store

    if not Path(universe_db).exists():
        raise ValueError(f"no universe database at {universe_db}: run `heartless universe build` first")
    store = Store(universe_db)
    try:
        res = research_universe(store, start, end, top)
    finally:
        store.close()
    if res["unranked"]:
        raise ValueError(f"months {', '.join(res['unranked'])} cannot be ranked from {universe_db} (their previous "
                         f"month's 1h data is not built): run `heartless universe build` over the window")
    have = stored_candle_symbols(candle_db, res["symbols"])
    return {"rule": f"top{top} by median daily quote volume of the previous month, listed >= {RESEARCH_MIN_AGE_DAYS}d "
                    f"(docs/UNIVERSE.md)", "research": res["symbols"],
            "symbols": sorted(s for s in res["symbols"] if s in have),
            "skipped": sorted(s for s in res["symbols"] if s not in have), "unranked": res["unranked"],
            "members": res["members"]}

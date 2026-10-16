"""Which symbols the bot trades (live) and which symbols research evaluates (historical), by rule instead of by name.

Live (select_live_universe): every USDT-margined perpetual that is trading, at least MIN_AGE_DAYS listed, not a
stablecoin, with 24h quote volume >= UNIVERSE_MIN_QUOTE_VOLUME, a tick no coarser than MAX_TICK_BPS of the price and,
when book tickers are supplied, a spread of at most MAX_SPREAD_BPS; the most liquid `size` of them trade. ALWAYS_INCLUDE
is a data subscription (BTC is the regime and market-context reference every alpha reads): those symbols are
subscribed even when they fail a liquidity rule, and they trade only when they pass the same rules as every other
symbol. select_universe() keeps the older list interface for app.py (see its docstring for the gap that remains
until app.py subscribes the reference symbols as data only).

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
    """List interface used by app.py, which TRADES every symbol of this list: the subscribed symbols of
    select_live_universe, `always` (ALWAYS_INCLUDE) first, capped at `size` (or the number of `always` symbols when
    larger).

    Known gap (not changed here: tests/test_fix_7_universe.py keeps the older age / volume exemption of ALWAYS_INCLUDE,
    and app.py has no data-only path for it yet): an `always` symbol that fails a liquidity or age rule is traded.
    With the default ALWAYS_INCLUDE=BTCUSDT that never happens in practice (BTC is the most liquid perpetual); a custom
    list can. The fix belongs with the app.py rewiring: subscribe `UniverseSelection.reference` as data only
    (context-only streams, BTC regime from that stream) and trade `UniverseSelection.trade`.
    `rules`: book_tickers, max_spread_bps, max_tick_bps, now (select_live_universe)."""
    sel = select_live_universe(infos, tickers24h, size, min_quote_volume, always, min_age_days, **rules)
    for s in sel.reference:
        if s in sel.rejected:
            log.warning("always-include %s is traded although it fails a live rule: %s", s, sel.rejected[s])
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
# A week or more without a trade, then trading again: a relisting (AERGOUSDT 2025-04 and PUMPUSDT 2025-07 came back as
# new contracts after their old ones settled; CVCUSDT returned in 2025-05 after 2.5 years). The exchange gives a
# relisted contract a new onboardDate, so the listing-age rule counts from the new first trade, as live does.
RELIST_GAP_DAYS = 7
# 1m coverage of a topN run (ROADMAP section 6, 2026-10-16; red-team F-H1). A member-month (or a membership span's
# warm-up) counts as covered when the candle database holds at least MIN_MONTH_COVERAGE of its expected minutes; a lab
# or discover run is refused (blocked_data) when more than MAX_MISSING_SHARE of its member-months or of its warm-ups
# are not covered, unless --allow-partial, which is never recorded in the trial ledger nor allowed in a FORWARD command.
MIN_MONTH_COVERAGE = 0.9
MAX_MISSING_SHARE = 0.02


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


def relistings(store, inventory: list[dict] | dict[str, dict] | None = None,
               gap_days: float = RELIST_GAP_DAYS) -> dict[str, list[int]]:
    """Per symbol, the first traded hour after every stretch of at least `gap_days` without a trade in the stored 1h
    blocks (ascending). The first trade of a symbol listed inside the stored months is its listing, not a relisting;
    one listed before them counts from the start of its first stored month (flat files there, then trades: it came
    back). Gaps across months come from the summary columns; a block decodes only when its traded days leave room for
    a gap inside the month."""
    import numpy as np

    from heartless.data.symbols_inventory import FUT_TABLE, decode_month, month_str

    inv = inventory if isinstance(inventory, dict) else {r["symbol"]: r for r in (
        inventory if inventory is not None else store.load_symbol_inventory())}
    gap = int(gap_days * MS_DAY)
    out: dict[str, list[int]] = {}
    last: dict[str, int | None] = {}
    for b in store.kline_block_index(FUT_TABLE):  # ordered by symbol, month
        sym = b["symbol"]
        if sym not in last:  # the symbol's first stored month
            first_month = (inv.get(sym) or {}).get("first_month")
            last[sym] = int(b["month"]) if first_month and first_month < month_str(b["month"]) else None
        if b["first_trade"] is None:
            continue
        ft, lt = int(b["first_trade"]), int(b["last_trade"])
        if last[sym] is not None and ft - last[sym] >= gap:
            out.setdefault(sym, []).append(ft)
        # a gap of G days between two traded hours leaves at least G - 1 whole calendar days without a trade
        if (lt // MS_DAY - ft // MS_DAY + 1) - int(b["days_traded"] or 0) >= gap_days - 1:
            blk = store.load_kline_blocks(FUT_TABLE, sym, b["month"], b["month"])
            if blk:
                d = decode_month(blk[0][1])
                t = d["open_time"][d["quote_volume"] > 0]
                out.setdefault(sym, []).extend(int(t[i + 1]) for i in np.flatnonzero(np.diff(t) >= gap))
        last[sym] = lt
    return {s: sorted(v) for s, v in out.items() if v}


def listed_as_of(row: dict, month_start: int, relisted: dict[str, list[int]] | None = None) -> int | None:
    """Listing time of an inventory row as the start of a month sees it: its latest relisting before the month when
    there is one (relistings()), else listed_at(row)."""
    base = listed_at(row)
    back = [t for t in (relisted or {}).get(row.get("symbol"), ()) if t < month_start]
    if back and base is not None:
        return max(base, back[-1])
    return base


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


def month_ranking(store, month: str, inventory: list[dict] | None = None, min_age_days: float = RESEARCH_MIN_AGE_DAYS,
                  relisted: dict[str, list[int]] | None = None) -> list[dict]:
    """Every eligible symbol ranked for calendar month `month` by the median daily quote volume of the previous month
    (days without bars count as 0; heartless/data/symbols_inventory.encode_month), most liquid first, ties by name.
    Only data before the month is used: the previous month's volumes and the listing time (a relisting counts from its
    new first trade, listed_as_of). Symbols listed less than `min_age_days` before the month starts, or without volume
    in the previous month, are not ranked. `relisted`: relistings(store) when the caller has it."""
    from heartless.data.symbols_inventory import FUT_TABLE

    inv = {r["symbol"]: r for r in (inventory if inventory is not None else store.load_symbol_inventory())}
    relisted = relistings(store, inv) if relisted is None else relisted
    m0 = _month_ms(month)
    ranked = []
    for b in store.kline_block_index(FUT_TABLE, month=_month_ms(prev_month(month))):
        row = inv.get(b["symbol"])
        if row is None or eligible_research(row) is not None or not b["n"] or not (b["med_dqv"] or 0) > 0:
            continue
        listed = listed_as_of(row, m0, relisted)
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
    relisted = relistings(store, inv)
    per_month: dict[str, list[str]] = {}
    unranked: list[str] = []
    members: dict[str, list[str]] = {}
    for mo in months_of(start, end):
        ranked = month_ranking(store, mo, inv, relisted=relisted)
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


def stored_candle_symbols(candle_db, symbols: list[str], start: int | None = None, end: int | None = None) -> set[str]:
    """Which of `symbols` have 1m candles in the candle database inside [start, end] (anywhere when not given), read
    through a read-only connection (one indexed lookup per symbol; the long-history database is never opened for
    writing). A member whose candles all lie outside the run's window would only come back as a lab error."""
    import sqlite3
    from pathlib import Path

    from heartless.core.store import readonly_uri

    if not Path(candle_db).exists():
        return set()
    con = sqlite3.connect(readonly_uri(candle_db), uri=True)  # E2: no -wal / -shm left beside a quiet database
    lo, hi = -(2**62) if start is None else int(start), 2**62 if end is None else int(end)
    try:
        have = set()
        for s in symbols:
            if con.execute("SELECT 1 FROM candles WHERE symbol=? AND open_time BETWEEN ? AND ? LIMIT 1",
                           (s, lo, hi)).fetchone():
                have.add(s)
        return have
    except sqlite3.OperationalError:  # no candles table
        return set()
    finally:
        con.close()


def _count_candles(con, symbol: str, a: int, b: int) -> int:
    try:
        return int(con.execute("SELECT COUNT(*) FROM candles WHERE symbol=? AND open_time BETWEEN ? AND ?",
                               (symbol, int(a), int(b))).fetchone()[0])
    except Exception:  # noqa: BLE001 - no candles table: nothing is covered
        return 0


def member_coverage(candle_db, members: dict[str, list[str]], start: int, end: int,
                    last_trade: dict[str, int | None] | None = None, listed: dict[str, list[int]] | None = None,
                    warmup_ms: int = 12 * MS_DAY, min_coverage: float = MIN_MONTH_COVERAGE) -> dict:
    """1m coverage of a topN run's member-months and of each membership span's warm-up, from the candle database
    (read-only, one indexed COUNT per member-month, about 2 ms each on the long-history database).

    A member-month is the month clipped to [start, end] and to the end of the symbol's last traded hour (a month after
    the delisting has nothing to trade and is not counted); it is covered when the stored 1m bars reach `min_coverage`
    of its minutes. A span's warm-up is the `warmup_ms` before the span start, from the latest listing at or before
    it on (`listed`: per symbol its listing and relisting times, ascending): the indicators of the span's first
    decisions need it. Why: "any candle in the window" let a run evaluate a fraction of
    its member set (TRAIN-F 1,779 of 1,920 member-months, VALID-F 137 of 360 on 2026-10-16) under the topN name."""
    import sqlite3
    from pathlib import Path

    from heartless.core.store import readonly_uri
    from heartless.data.symbols_inventory import _next_month_ms

    MS_MIN = 60_000
    con = sqlite3.connect(readonly_uri(candle_db), uri=True) if Path(candle_db).exists() else None
    months_total = 0
    missing: list[dict] = []
    spans_total = 0
    warm_missing: list[dict] = []
    try:
        for sym in sorted(members):
            stop = last_trade.get(sym) if last_trade else None
            cap = int(stop) + 3_600_000 - 1 if stop is not None else int(end)
            for mo in sorted(set(members[sym])):
                a, b = max(_month_ms(mo), int(start)), min(_next_month_ms(mo) - 1, int(end), cap)
                if b < a:
                    continue
                need = (b - a) // MS_MIN + 1
                have = _count_candles(con, sym, a, b) if con is not None else 0
                months_total += 1
                if have < min_coverage * need:
                    missing.append({"symbol": sym, "month": mo, "minutes": have, "expected": need})
            for a, b in membership_windows({sym: members[sym]}, start, min(int(end), cap)).get(sym, []):
                lo = a - int(warmup_ms)
                back = [int(t) for t in (listed or {}).get(sym) or () if t is not None and t <= a]
                if back:
                    lo = max(lo, back[-1])
                if lo >= a or warmup_ms <= 0:
                    continue
                need = (a - lo) // MS_MIN
                have = _count_candles(con, sym, lo, a - 1) if con is not None else 0
                spans_total += 1
                if have < min_coverage * need:
                    warm_missing.append({"symbol": sym, "span_start": a, "minutes": have, "expected": need})
    finally:
        if con is not None:
            con.close()
    share = len(missing) / months_total if months_total else 0.0
    wshare = len(warm_missing) / spans_total if spans_total else 0.0
    return {"rule": f"a member-month (or a span's {warmup_ms / MS_DAY:g}-day warm-up) is covered with >= "
                    f"{min_coverage:.0%} of its minutes stored; refused above {MAX_MISSING_SHARE:.0%} missing "
                    f"(ROADMAP section 6, 2026-10-16)",
            "member_months": months_total, "missing_member_months": missing, "missing_share": round(share, 5),
            "warmups": spans_total, "missing_warmups": warm_missing, "missing_warmup_share": round(wshare, 5),
            "max_missing_share": MAX_MISSING_SHARE}


def coverage_problem(coverage: dict | None, max_share: float = MAX_MISSING_SHARE) -> str | None:
    """Why a topN run's member set is too incomplete to stand for the rule (blocked_data), or None."""
    if not coverage:
        return None
    out = []
    if coverage["missing_share"] > max_share:
        out.append(f"{len(coverage['missing_member_months'])} of {coverage['member_months']} member-months "
                   f"({coverage['missing_share']:.1%}) lack 1m candles")
    if coverage["missing_warmup_share"] > max_share:
        out.append(f"{len(coverage['missing_warmups'])} of {coverage['warmups']} membership spans "
                   f"({coverage['missing_warmup_share']:.1%}) lack their 1m warm-up")
    return "; ".join(out) + f" (at most {max_share:.0%} may be missing)" if out else None


def membership_windows(members: dict[str, list[str]], start: int | None = None,
                       end: int | None = None) -> dict[str, list[tuple[int, int]]]:
    """Each member's months as contiguous [first ms, last ms] spans, clipped to [start, end]: the hours in which the
    point-in-time rule had the symbol in the universe."""
    from heartless.data.symbols_inventory import _next_month_ms

    out: dict[str, list[tuple[int, int]]] = {}
    for sym, months in members.items():
        spans: list[tuple[int, int]] = []
        for mo in sorted(set(months)):
            a, b = _month_ms(mo), _next_month_ms(mo) - 1
            if spans and spans[-1][1] == a - 1:
                spans[-1] = (spans[-1][0], b)
            else:
                spans.append((a, b))
        clipped = [(max(a, start if start is not None else a), min(b, end if end is not None else b)) for a, b in spans]
        out[sym] = [(a, b) for a, b in clipped if a <= b]
    return out


def member_mask(times, spans: list[tuple[int, int]]):
    """Boolean mask of `times` (ms, ascending or not) that fall inside one of `spans` ([first, last] inclusive, sorted
    and disjoint as membership_windows gives them): the vectorised member test of the hourly evaluator and discover."""
    import numpy as np

    t = np.asarray(times, dtype=np.int64)
    if not spans or not len(t):
        return np.zeros(len(t), dtype=bool)
    lo = np.array([a for a, _ in spans], dtype=np.int64)
    hi = np.array([b for _, b in spans], dtype=np.int64)
    k = np.searchsorted(lo, t, side="right") - 1
    ok = k >= 0
    ok[ok] &= t[ok] <= hi[k[ok]]
    return ok


def member_months(windows: dict[str, list[tuple[int, int]]]) -> dict[str, int]:
    """Calendar months covered by each symbol's membership spans (a clipped first / last month counts once)."""
    out = {}
    for sym, spans in windows.items():
        months: set[str] = set()
        for a, b in spans:
            months.update(months_of(int(a), int(b)))
        out[sym] = len(months)
    return out


def delisting_closes(store, symbols: list[str] | None = None, start: int | None = None,
                     end: int | None = None) -> dict[str, int]:
    """Per delisted symbol, the last instant of its last traded hour (open of that hour + 1 h - 1 ms): where a position
    still open is force-closed (ROADMAP 7.4: that hour's close, 5x the nominal slippage, 10x under stress). Delisted =
    no trade in the last ACTIVE_SLACK_MS of the stored data (symbols_inventory.symbol_status); only closes inside
    [start, end] are returned. A pause followed by a relisting is not a delisting here (the 1h / 1m data simply has a
    gap); the announcement date is not in the data, so nothing blocks entries before the delisting."""
    from heartless.data.symbols_inventory import ACTIVE_SLACK_MS, MS_H, data_end

    stop = data_end(store)
    if stop is None:
        return {}
    wanted = None if symbols is None else set(symbols)
    out = {}
    for r in store.load_symbol_inventory():
        sym, last = r["symbol"], r.get("last_trade")
        if last is None or (wanted is not None and sym not in wanted) or last >= stop - ACTIVE_SLACK_MS:
            continue
        close = int(last) + MS_H - 1
        if (start is None or close >= start) and (end is None or close <= end):
            out[sym] = close
    return out


def member_trades(trades: list[dict], windows: dict[str, list[tuple[int, int]]]) -> list[dict]:
    """The trades entered while their symbol was a member (entry_time inside one of its membership_windows).

    Why: a topN universe is the union of every month's members, and lab / discover evaluate each of them over the whole
    window. A symbol that became top N because of a surge in month m-1 is then also evaluated in m-1 and before, i.e.
    selected on the outcome it is evaluated on. Only the trades this keeps are free of that look-ahead."""
    return [t for t in trades
            if any(a <= int(t.get("entry_time") or 0) <= b for a, b in windows.get(t.get("symbol"), ()))]


def member_windows_for(universe_db, symbols: list[str], start: int, end: int, top: int) -> dict:
    """Membership of a FIXED symbol list over another window by the same monthly rule: {"windows": spans of each of
    `symbols` that is top-`top` in a month of [start, end] (the others get no span), "delisted": their delisting
    closes in it, and before it for a symbol that still has a span}. discover uses it for VALID: the symbol list is
    frozen at the end of TRAIN (no VALID-period liquidity chooses symbols), and each frozen symbol counts only in the
    VALID months it is a member (each month ranked on the month before it, so this is point-in-time)."""
    from pathlib import Path

    from heartless.core.store import Store

    if not Path(universe_db).exists():
        raise ValueError(f"no universe database at {universe_db}: run `heartless universe build` first")
    store = Store(universe_db, readonly=True)  # a shared research dataset: only read (E2)
    try:
        res = research_universe(store, start, end, top)
        delisted = delisting_closes(store, list(symbols), None, end)
    finally:
        store.close()
    if res["unranked"]:
        raise ValueError(f"months {', '.join(res['unranked'])} cannot be ranked from {universe_db} (their previous "
                         f"month's 1h data is not built): run `heartless universe build` over the window")
    members = {s: res["members"].get(s, []) for s in symbols}
    windows = membership_windows(members, start, end)
    # a delisting before the window matters only for a symbol that is still a member in it (month m ranks on m-1, so a
    # symbol delisted late in m-1 can rank): its spans must not trade the flat files published after the last trade
    return {"windows": windows, "delisted": {s: v for s, v in delisted.items() if v >= start or windows.get(s)}}


def resolve_for_run(universe_db, candle_db, start: int, end: int, top: int, coverage: bool = False,
                    warmup_ms: int = 12 * MS_DAY) -> dict:
    """The topN research universe of [start, end] for a lab / discovery run over `candle_db`: "research" = every
    member (research_universe), "symbols" = those with stored 1m candles inside the window (what the run evaluates),
    "skipped" = the other members, to be reported (fetch them with `heartless fetch --start ... --symbols ...`),
    "windows" = each evaluated member's membership spans (membership_windows; member_trades filters a run's trades by
    them). `coverage`: also "coverage" = member_coverage over every member (the lab / discover refusal reads it via
    coverage_problem). ValueError when the universe database is missing or a month of the window cannot be ranked
    yet."""
    from pathlib import Path

    from heartless.core.store import Store

    if not Path(universe_db).exists():
        raise ValueError(f"no universe database at {universe_db}: run `heartless universe build` first")
    store = Store(universe_db, readonly=True)  # a shared research dataset: only read (E2)
    try:
        res = research_universe(store, start, end, top)
    finally:
        store.close()
    if res["unranked"]:
        raise ValueError(f"months {', '.join(res['unranked'])} cannot be ranked from {universe_db} (their previous "
                         f"month's 1h data is not built): run `heartless universe build` over the window")
    have = stored_candle_symbols(candle_db, res["symbols"], start, end)
    symbols = sorted(s for s in res["symbols"] if s in have)
    windows = membership_windows({s: res["members"][s] for s in symbols}, start, end)
    skipped = sorted(s for s in res["symbols"] if s not in have)
    store = Store(universe_db, readonly=True)
    try:
        # every member's delisting up to the window end, also one before the window start (a member month can follow
        # the last traded hour: month m ranks on m-1's volume): the lab runs no job after it (lab.evaluate)
        delisted = delisting_closes(store, res["symbols"], None, end)
        listed: dict[str, list[int]] = {}
        if coverage:
            inv = {r["symbol"]: r for r in store.load_symbol_inventory()}
            relisted = relistings(store, inv)
            for s in res["symbols"]:
                first = listed_at(inv[s]) if s in inv else None
                listed[s] = sorted(([first] if first is not None else []) + list(relisted.get(s, ())))
    finally:
        store.close()
    cov = (member_coverage(candle_db, {s: res["members"][s] for s in res["symbols"]}, start, end, res["last_trade"],
                           listed, warmup_ms) if coverage else None)
    months = member_months(windows)
    skipped_months = member_months(membership_windows({s: res["members"][s] for s in skipped}, start, end))
    return {"rule": f"top{top} by median daily quote volume of the previous month, listed >= {RESEARCH_MIN_AGE_DAYS}d "
                    f"(docs/UNIVERSE.md)", "research": res["symbols"], "symbols": symbols,
            "skipped": skipped, "unranked": res["unranked"],
            "members": res["members"], "windows": windows,
            # member-months of the evaluated symbols and of the skipped ones (no 1m candles in the window), and the
            # members whose trading ended by the window end (delisting close, docs/ROADMAP.md 7.4)
            "member_months": months, "member_months_total": sum(months.values()),
            "skipped_member_months": sum(skipped_months.values()), "delisted": delisted,
            **({"coverage": cov} if cov is not None else {}),
            "member_share": round(sum(b - a + 1 for w in windows.values() for a, b in w)
                                  / max(1, len(symbols) * (int(end) - int(start) + 1)), 4)}

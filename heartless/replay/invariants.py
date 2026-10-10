"""Replay invariants (docs/ROADMAP.md P5), checked every virtual minute and once more after the final flatten.

Every check compares two independently kept books that must agree, never a value with itself:

1. positions:   each engine's open positions == its account's positions (symbol, side, quantity within half a step);
                a pending entry holds nothing on the account.
2. stops:       every open position's `sl_algo_id` is a working STOP_MARKET on its account, on the closing side,
                covering the position, triggering at the engine's stop.
3. orphans:     every working order/algo on an account belongs to a tracked position (its bracket ids) or to the
                working entry order of a pending/partially filled entry; at most one stop per symbol.
4. ledger:      per account, wallet change == realized - fees + funding (the simulator's own counters), and the same
                wallet change == stored trade PnL + open positions' realized - fees + funding (the engine's books).
5. exceptions:  no unhandled exception in any task or callback (loop exception handler), no error-level log record,
                no loop deadlock.
6. pnl_reports: the PnL the bot reports (daily report, /pnl, the "today" line of close messages) equals the sum over
                the stored trades of that reporting day (settings.timezone; the replay uses UTC), parsed from the
                captured Telegram text.
7. memory:      RSS (and optionally tracemalloc) sampled every virtual hour; growth after warm-up and its slope bounded.
8. trades:      stored trades are complete (exit price/time, pnl, r_multiple finite; every CLOSED position has its trade)
                and after the final flatten no position is left open in the store, the engines or the accounts.
9. network:     no real network connection was attempted.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from heartless.core.models import PositionStatus
from heartless.exchange.paper import PaperAccount
from heartless.util.timeutil import MS_DAY, fmt_ts, local_day_start, now_ms

log = logging.getLogger(__name__)

CHECKS = {
    "positions": "engine open positions == account positions",
    "stops": "every open position has its protective stop working",
    "orphans": "no orphan working orders / algos",
    "ledger": "balance change == realized - fees + funding (account and engine books)",
    "exceptions": "no unhandled exception, error log or deadlock",
    "pnl_reports": "reported daily PnL == stored trades of that day",
    "memory": "memory growth bounded",
    "trades": "stored trades complete, nothing open after the final flatten",
    "network": "no network access",
}
_OPEN = (PositionStatus.OPEN, PositionStatus.CLOSING)
_USD = r"([+\-]?[\d,]+\.\d+) USDT"
_RE_DAILY_TODAY = re.compile(r"오늘 (\d+)건 · 실현 <b>" + _USD + "</b>")
_RE_DAILY_WEEK = re.compile(r"7일 (\d+)건 · 실현 " + _USD)
_RE_PNL_HEAD = re.compile(r"<b>(오늘|최근 7일|최근 30일|전체) 손익</b> \(([^)]*)\)")
_RE_PNL_NET = re.compile(r"실현 <b>" + _USD + "</b>")
_RE_PNL_N = re.compile(r"\n(\d+)건 · 승률")
_RE_CLOSED_TODAY = re.compile(r"오늘 실현손익 " + _USD)
_PERIODS = {"오늘": "today", "최근 7일": "week", "최근 30일": "month", "전체": "all"}


@dataclass
class Violation:
    ts: int
    check: str
    engine: str
    detail: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = fmt_ts(self.ts)
        return d


def _usd(text: str) -> float:
    return float(text.replace(",", ""))


def rss_mb() -> float:
    """Resident set size of this process in MB (Linux /proc; 0 when unavailable)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        import os

        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, IndexError):
        return 0.0


class InvariantChecker:
    def __init__(self, app, tolerance: float = 1e-6, keep: int = 50):
        self.app = app
        self.tol = tolerance
        self.keep = keep  # violations stored in full per check (all are counted)
        self.runs = {k: 0 for k in CHECKS}
        self.counts = {k: 0 for k in CHECKS}
        self.samples: list[Violation] = []
        self._kept: Counter = Counter()
        self._baseline: dict[str, tuple] = {}
        self.dangling_refs = 0  # bracket ids an open position still names but its account no longer works (info)
        self.memory: list[dict] = []
        self._trades_checked_until = -1
        self._seen_trades: set[str] = set()

    # --- bookkeeping -------------------------------------------------------------------------------
    def violation(self, check: str, detail: str, engine: str = "", ts: int | None = None) -> None:
        self.counts[check] += 1
        if self._kept[check] < self.keep:
            self._kept[check] += 1
            v = Violation(now_ms() if ts is None else ts, check, engine, detail)
            self.samples.append(v)
            log.warning("INVARIANT %s [%s] %s", check, engine or "-", detail)

    @property
    def total_violations(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> dict:
        return {"checks": {k: {"runs": self.runs[k], "violations": self.counts[k], "what": CHECKS[k]} for k in CHECKS},
                "violations_total": self.total_violations, "violations": [v.to_dict() for v in self.samples],
                "dangling_bracket_refs": self.dangling_refs}

    # --- every minute ------------------------------------------------------------------------------
    async def check_minute(self) -> None:
        # the exception / network watches are armed all the time; a minute counts as one pass of each
        self.runs["exceptions"] += 1
        self.runs["network"] += 1
        for name, eng in list(self.app.engines.items()):
            acc = eng.account
            if not isinstance(acc, PaperAccount):
                continue
            algos = await acc.open_algo_orders()
            orders = await acc.open_orders()
            self._positions(name, eng, acc)
            self._stops(name, eng, algos)
            self._orphans(name, eng, algos, orders)
            self._ledger(name, eng, acc)

    def _step(self, sym: str) -> float:
        info = self.app.symbols.get(sym)
        return info.step_size if info else 0.0

    def _positions(self, name: str, eng, acc: PaperAccount) -> None:
        self.runs["positions"] += 1
        held = {s: p for s, p in acc.positions.items() if abs(p.qty) > 0}
        tracked = {}
        for sym, p in eng.positions.items():
            if p.symbol != sym:
                self.violation("positions", f"position keyed {sym} holds {p.symbol}", name)
            if p.status in _OPEN and p.qty > 0:
                tracked[sym] = p
            elif p.status is PositionStatus.PENDING:
                if sym in held:
                    self.violation("positions", f"{sym} entry is PENDING but the account holds {held[sym].qty:g}", name)
            elif p.status in (PositionStatus.CLOSED, PositionStatus.CANCELLED):
                self.violation("positions", f"{sym} {p.status.value} position is still tracked as live", name)
            elif p.status in _OPEN and p.qty <= 0:
                self.violation("positions", f"{sym} {p.status.value} position with qty {p.qty:g}", name)
        for sym in sorted(set(held) | set(tracked)):
            p, pp = tracked.get(sym), held.get(sym)
            if p is None:
                if sym not in eng.positions:
                    self.violation("positions", f"account holds {pp.qty:+g} {sym}, engine tracks nothing", name)
                continue
            if pp is None:
                self.violation("positions", f"engine {p.status.value} {p.side.value} {p.qty:g} {sym}, account is flat", name)
                continue
            if (pp.qty > 0) != (p.side.sign > 0):
                self.violation("positions", f"{sym} side: engine {p.side.value}, account {pp.qty:+g}", name)
            elif abs(abs(pp.qty) - p.qty) > self._step(sym) * 0.5 + 1e-12:
                self.violation("positions", f"{sym} qty: engine {p.qty:g}, account {abs(pp.qty):g}", name)

    def _stops(self, name: str, eng, algos: list[dict]) -> None:
        self.runs["stops"] += 1
        by_id = {str(a.get("algoId")): a for a in algos}
        for sym, p in eng.positions.items():
            if p.status is not PositionStatus.OPEN or p.qty <= 0:
                continue
            a = by_id.get(str(p.sl_algo_id)) if p.sl_algo_id else None
            info = self.app.symbols.get(sym)
            tick = info.tick_size if info else 0.0
            if a is None:
                self.violation("stops", f"{sym} {p.side.value} {p.qty:g} has no working stop (sl_algo_id={p.sl_algo_id!r})",
                               name)
            elif a.get("symbol") != sym or a.get("orderType") != "STOP_MARKET" or a.get("side") != p.side.close_order_side:
                self.violation("stops", f"{sym} sl_algo_id {p.sl_algo_id} is {a.get('orderType')} {a.get('side')} on "
                                        f"{a.get('symbol')}", name)
            elif not a.get("closePosition") and float(a.get("quantity") or 0.0) < p.qty - self._step(sym) * 0.5 - 1e-12:
                self.violation("stops", f"{sym} stop covers {a.get('quantity')} of {p.qty:g}", name)
            elif abs(float(a.get("triggerPrice") or 0.0) - p.stop) > tick * 0.5 + 1e-9:
                self.violation("stops", f"{sym} stop triggers at {a.get('triggerPrice')}, engine stop {p.stop}", name)

    def _orphans(self, name: str, eng, algos: list[dict], orders: list[dict]) -> None:
        self.runs["orphans"] += 1
        working = {str(a.get("algoId")) for a in algos}
        for a in algos:
            sym, aid = a.get("symbol"), str(a.get("algoId"))
            p = eng.positions.get(sym)
            owned = p is not None and p.status in _OPEN and aid in {str(x) for x in (
                p.sl_algo_id, p.tp_algo_id, p.extra.get("tp1_algo_id")) if x}
            if not owned:
                what = "no position" if p is None else f"{p.status.value} position {p.id}"
                self.violation("orphans", f"working {a.get('orderType')} {a.get('side')} {sym} {aid} "
                                          f"@{a.get('triggerPrice')} belongs to nothing ({what})", name)
        stops = Counter(a.get("symbol") for a in algos if a.get("orderType") == "STOP_MARKET")
        for sym, n in stops.items():
            if n > 1:
                self.violation("orphans", f"{n} working stops on {sym}", name)
        for o in orders:
            sym, cid = o.get("symbol"), o.get("clientOrderId")
            p = eng.positions.get(sym)
            if o.get("reduceOnly"):
                self.violation("orphans", f"resting reduce-only order {cid} on {sym}", name)
            elif p is None or p.status not in (PositionStatus.PENDING, PositionStatus.OPEN):
                self.violation("orphans", f"working entry order {cid} on {sym} without a pending/open position", name)
            elif cid != p.entry_client_id:
                self.violation("orphans", f"stale entry order {cid} still working on {sym} (current "
                                          f"{p.entry_client_id})", name)
        for sym, p in eng.positions.items():
            if p.status is PositionStatus.OPEN:
                for aid in (p.tp_algo_id, p.extra.get("tp1_algo_id")):
                    if aid and str(aid) not in working:
                        self.dangling_refs += 1

    def _ledger(self, name: str, eng, acc: PaperAccount) -> None:
        self.runs["ledger"] += 1
        open_net = sum(p.realized - p.fees + p.funding for p in eng.positions.values() if p.status in _OPEN)
        trades_net, _ = self._trades_sum(name)
        base = self._baseline.get(name)
        if base is None or base[0] != id(acc):
            # first sight of this account (a fresh challenger reuses a slot name): its starting books
            base = (id(acc), acc.wallet, acc.realized, acc.fees_paid, acc.funding_paid, trades_net, open_net)
            self._baseline[name] = base
        _, w0, r0, f0, fu0, t0, o0 = base
        scale = max(1.0, abs(acc.initial_balance))
        dw = acc.wallet - w0
        own = (acc.realized - r0) - (acc.fees_paid - f0) + (acc.funding_paid - fu0)
        if abs(dw - own) > self.tol * scale:
            self.violation("ledger", f"wallet change {dw:+.8f} != realized - fees + funding {own:+.8f}", name)
        books = (trades_net - t0) + (open_net - o0)
        if abs(dw - books) > self.tol * scale:
            self.violation("ledger", f"wallet change {dw:+.8f} != stored trades + open positions {books:+.8f} "
                                     f"(diff {dw - books:+.8f})", name)

    def _trades_sum(self, engine: str, since: int | None = None, until: int | None = None) -> tuple[float, int]:
        sql = "SELECT COALESCE(SUM(pnl), 0) s, COUNT(*) n FROM trades WHERE engine=?"
        params: list = [engine]
        if since is not None:
            sql += " AND exit_time>=?"
            params.append(since)
        if until is not None:
            sql += " AND exit_time<=?"
            params.append(until)
        r = self.app.store.query(sql, params)[0]
        return float(r["s"]), int(r["n"])

    # --- exceptions / network ----------------------------------------------------------------------
    def loop_exception(self, context: dict) -> None:
        exc = context.get("exception")
        what = context.get("message", "")
        task = context.get("task") or context.get("future")
        name = task.get_name() if hasattr(task, "get_name") else ""
        detail = f"{what}{' in ' + name if name else ''}: {type(exc).__name__}: {exc}" if exc else what
        self.violation("exceptions", f"unhandled: {detail}")

    def error_log(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            msg += f" [{type(record.exc_info[1]).__name__}: {record.exc_info[1]}]"
        self.violation("exceptions", f"{record.levelname} {record.name}: {msg[:400]}")

    def network(self, msg: str) -> None:
        self.violation("network", msg)

    # --- reported PnL ------------------------------------------------------------------------------
    def on_message(self, m) -> None:
        """Telegram capture listener: compare every PnL figure the bot reports with the stored trades."""
        try:
            if m.topic == "daily_report":
                self._check_daily(m)
            elif m.topic == "cmd:/pnl":
                self._check_pnl_cmd(m)
            elif m.topic == "position_closed":
                self._check_closed(m)
        except Exception as e:  # noqa: BLE001 - a parsing problem is itself a finding
            self.violation("pnl_reports", f"{m.topic} message could not be checked ({type(e).__name__}: {e}): "
                                          f"{m.text[:200]!r}", ts=m.ts)

    def _compare(self, what: str, engine: str, shown_n: int | None, shown: float, since: int | None, ts: int) -> None:
        self.runs["pnl_reports"] += 1
        exp, n = self._trades_sum(engine, since, ts)
        if (shown_n is not None and shown_n != n) or abs(shown - exp) > 0.0051:
            self.violation("pnl_reports", f"{what}: shown {shown_n if shown_n is not None else '-'} trades "
                                          f"{shown:+.2f}, stored {n} trades {exp:+.4f} (since "
                                          f"{fmt_ts(since) if since else 'start'})", engine, ts)

    def _primary(self) -> str:
        prim = self.app.primary_engine
        return prim.name if prim else "paper"

    def _day_start(self, ts: int) -> int:
        return local_day_start(ts, self.app.s.timezone)

    def _check_daily(self, m) -> None:
        a, b = _RE_DAILY_TODAY.search(m.text), _RE_DAILY_WEEK.search(m.text)
        if not a or not b:
            raise ValueError("daily report layout not recognised")
        eng = self._primary()
        self._compare("daily report today", eng, int(a.group(1)), _usd(a.group(2)), self._day_start(m.ts), m.ts)
        self._compare("daily report 7d", eng, int(b.group(1)), _usd(b.group(2)), m.ts - 7 * MS_DAY, m.ts)

    def _check_pnl_cmd(self, m) -> None:
        head, net, n = _RE_PNL_HEAD.search(m.text), _RE_PNL_NET.search(m.text), _RE_PNL_N.search(m.text)
        if not head or not net or not n:
            raise ValueError("/pnl layout not recognised")
        period = _PERIODS[head.group(1)]
        since = {"today": self._day_start(m.ts), "week": m.ts - 7 * MS_DAY, "month": m.ts - 30 * MS_DAY,
                 "all": None}[period]
        self._compare(f"/pnl {period}", head.group(2), int(n.group(1)), _usd(net.group(1)), since, m.ts)

    def _check_closed(self, m) -> None:
        a = _RE_CLOSED_TODAY.search(m.text)
        if not a:
            raise ValueError("close message layout not recognised")
        label = m.text.split(" · ", 1)[0]
        eng = "live" if "LIVE" in label else ("paper" if "PAPER" in label else label.split()[-1])
        self._compare("close message 'today'", eng, None, _usd(a.group(1)), self._day_start(m.ts), m.ts)

    # --- memory ------------------------------------------------------------------------------------
    def sample_memory(self, tracemalloc_on: bool = False) -> None:
        app = self.app
        row = {"ts": now_ms(), "rss_mb": round(rss_mb(), 1),
               "candle_rows": int(sum(ca.n for ca in app.candles.values())),
               "candle_capacity": int(sum(getattr(ca, "_cap", 0) for ca in app.candles.values())),
               "bus_history": len(app.bus.history),
               "engine_closed": int(sum(len(e.closed) for e in app.engines.values())),
               "metrics_rows": int(sum(len(m) for m in app.metrics_series.values())),
               "telegram_queue": app.telegram._queue.qsize() if app.telegram is not None else 0}
        if tracemalloc_on:
            import tracemalloc

            row["traced_mb"] = round(tracemalloc.get_traced_memory()[0] / 1e6, 1)
        self.memory.append(row)
        self.runs["memory"] += 1

    def check_memory(self, limit_mb: float, slope_mb_per_day: float, warmup_ms: int) -> dict:
        """Growth of RSS from the first sample after `warmup_ms` to the last, and its least-squares slope."""
        if len(self.memory) < 3:
            return {"samples": len(self.memory)}
        t0 = self.memory[0]["ts"] + warmup_ms
        tail = [r for r in self.memory if r["ts"] >= t0] or self.memory[-2:]
        xs = [(r["ts"] - tail[0]["ts"]) / MS_DAY for r in tail]
        ys = [r["rss_mb"] for r in tail]
        growth = ys[-1] - ys[0]
        slope = 0.0
        if len(tail) >= 3 and xs[-1] > 0:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        out = {"samples": len(self.memory), "rss_start_mb": self.memory[0]["rss_mb"], "rss_after_warmup_mb": ys[0],
               "rss_end_mb": ys[-1], "rss_peak_mb": max(r["rss_mb"] for r in self.memory),
               "growth_after_warmup_mb": round(growth, 1), "slope_mb_per_day": round(slope, 2),
               "limit_mb": limit_mb, "slope_limit_mb_per_day": slope_mb_per_day}
        if growth > limit_mb:
            self.violation("memory", f"RSS grew {growth:.0f} MB after warm-up (limit {limit_mb:.0f} MB)")
        if xs[-1] >= 2 and slope > slope_mb_per_day:
            self.violation("memory", f"RSS slope {slope:.1f} MB/day after warm-up (limit {slope_mb_per_day:.1f})")
        return out

    # --- trades ------------------------------------------------------------------------------------
    def check_trades(self, final: bool = False) -> None:
        """New stored trades must be complete; CLOSED positions must have their trade. After the final flatten
        nothing may be left open anywhere."""
        self.runs["trades"] += 1
        store = self.app.store
        rows = store.query("SELECT position_id, engine, symbol, entry_time, exit_time, exit_price, pnl, r_multiple, qty "
                           "FROM trades WHERE exit_time>=? OR exit_time IS NULL", (self._trades_checked_until,))
        for r in rows:
            if r["position_id"] in self._seen_trades:
                continue
            self._seen_trades.add(r["position_id"])
            bad = [k for k in ("exit_time", "exit_price", "pnl", "r_multiple", "qty")
                   if r[k] is None or not math.isfinite(float(r[k]))]
            if not bad and (r["exit_price"] <= 0 or r["qty"] <= 0 or r["exit_time"] < (r["entry_time"] or 0)):
                bad = ["exit_price/qty/exit_time range"]
            if bad:
                self.violation("trades", f"trade {r['position_id']} {r['symbol']} incomplete: {', '.join(bad)}",
                               r["engine"])
        self._trades_checked_until = max([self._trades_checked_until] + [int(r["exit_time"] or 0) for r in rows])
        for r in store.query("SELECT p.id, p.engine, p.symbol FROM positions p LEFT JOIN trades t ON t.position_id=p.id "
                             "WHERE p.status='CLOSED' AND t.position_id IS NULL"):
            self.violation("trades", f"CLOSED position {r['id']} {r['symbol']} has no stored trade", r["engine"])
        if not final:
            return
        for r in store.query("SELECT id, engine, symbol, status FROM positions WHERE status IN ('OPEN','PENDING','CLOSING')"):
            self.violation("trades", f"position {r['id']} {r['symbol']} left {r['status']} in the store after the "
                                     f"final flatten", r["engine"])
        for name, eng in self.app.engines.items():
            if eng.positions:
                self.violation("trades", f"engine still tracks {sorted(eng.positions)} after the final flatten", name)
            acc = eng.account
            if isinstance(acc, PaperAccount):
                held = [s for s, p in acc.positions.items() if abs(p.qty) > 0]
                if held or acc.orders or acc.algos:
                    self.violation("trades", f"account not flat after the final flatten: positions {held}, "
                                             f"{len(acc.orders)} orders, {len(acc.algos)} algos", name)

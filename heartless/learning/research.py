"""Self-improvement loop: champion/challenger parameter management.

* Every `research_interval_minutes` a worker process runs walk-forward optimisation per alpha.
* Improvements become *challengers*: full parameter sets traded by dedicated paper engines on the
  live data feed, side by side with the champion.
* A challenger that beats the champion on real-time paper results (enough trades, better
  objective, no worse drawdown) is promoted; the live engine hot-swaps to the new champion.
* In paper mode the champion's live-feed record is also checked against the graduation criteria,
  and the owner is offered a one-tap "go live" on Telegram.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from heartless.execution.stats import objective, summarize
from heartless.learning.optimizer import run_research_cycle
from heartless.strategy.params import StrategyParams
from heartless.util.ids import short_id
from heartless.util.timeutil import MS_DAY, now_ms

log = logging.getLogger(__name__)


def _max_dd_r(trades: list[dict]) -> float:
    """Max drawdown of cumulative R multiples ordered by exit time.

    Scale-free, unlike the USDT `max_dd` from summarize(): challenger paper accounts restart at
    paper_initial_balance while the champion wallet carries its full history, and position sizes scale
    with equity, so absolute drawdowns of the two accounts are not comparable.
    """
    cum = peak = dd = 0.0
    for t in sorted(trades, key=lambda t: t.get("exit_time") or 0):
        cum += float(t.get("r_multiple") or 0.0)
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


@dataclass
class ChallengerSlot:
    name: str
    params: StrategyParams
    started: int
    source: str = ""


class ResearchManager:
    def __init__(self, app):
        self.app = app
        self.s = app.s
        self.store = app.store
        self.pool: ProcessPoolExecutor | None = None
        self.last_cycle_ts: int = int(self.store.get("research.last_cycle", 0) or 0)
        self.last_summary: dict = self.store.get("research.last_summary", {}) or {}
        self.last_promotion_check: int = 0
        self.graduation_notified_day: int = int(self.store.get("research.graduation_day", 0) or 0)
        self.running = False
        self.history: list[dict] = []
        self.research_failures: int = 0
        self.last_error: str = ""

    # --- champion persistence ------------------------------------------------------------------
    def load_champion(self) -> StrategyParams:
        rows = self.store.load_params_versions(role="champion", limit=1)
        if rows:
            p = StrategyParams.from_dict(rows[0]["params"])
            p.version = rows[0]["id"]
            p.note = rows[0].get("note", "")
            return p
        p = StrategyParams.default()
        self.store.save_params_version(p.version, p.created, "champion", "default", p.note, p.to_dict())
        return p

    def load_challengers(self) -> list[ChallengerSlot]:
        slots: list[ChallengerSlot] = []
        for row in self.store.load_params_versions(role="challenger", limit=self.s.challengers):
            p = StrategyParams.from_dict(row["params"])
            p.version = row["id"]
            slots.append(ChallengerSlot(name=row["metrics"].get("slot", f"challenger-{len(slots) + 1}"), params=p,
                                        started=row["metrics"].get("started", row["created"]), source=row.get("note", "")))
        return slots

    # --- research cycle ------------------------------------------------------------------------
    def due(self, now: int) -> bool:
        return (now - self.last_cycle_ts) >= self.s.research_interval_minutes * 60_000

    async def cycle(self, force: bool = False) -> dict | None:
        if self.running:
            return None
        now = now_ms()
        if not force and not self.due(now):
            return None
        self.running = True
        self.last_cycle_ts = now
        self.store.set("research.last_cycle", now)
        try:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            loop = asyncio.get_running_loop()
            settings_dict = {k: v for k, v in self.s.model_dump(by_alias=True).items()
                             if k not in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY")}
            settings_dict = {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in settings_dict.items()}
            symbols_dict = {k: asdict(v) for k, v in self.app.symbols.items() if k in self.app.universe}
            await self.app.emit("research_started", {"message": "퀀트 리서치 사이클 시작 (워크포워드 최적화)"})
            result = await loop.run_in_executor(self.pool, run_research_cycle, str(self.s.db_path), settings_dict,
                                                self.app.params.to_dict(), symbols_dict, self.s.research_lookback_days,
                                                self.s.research_candidates, now % 100_000, None,
                                                self.store.get("advisor.seeds", {}) or {})
        except Exception as e:  # noqa: BLE001
            log.exception("research cycle failed")
            # A crashed worker (OOM kill, os._exit) leaves the executor permanently broken: every later
            # submit() raises BrokenProcessPool immediately. Discard it so the next cycle recreates one.
            if self.pool is not None:
                try:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001
                    log.debug("research pool shutdown after failure raised", exc_info=True)
                self.pool = None
            self.research_failures += 1
            self.last_error = f"{now}: {e}"
            await self.app.emit("error", {"message": f"리서치 사이클 실패: {e}"})
            self.running = False
            return None
        self.running = False
        if "error" in result:
            await self.app.emit("research_done", {"message": f"리서치 건너뜀: {result['error']}", "result": result})
            return result
        self.last_summary = result
        self.store.set("research.last_summary", result)
        self.store.save_research_run(now, "all", result)
        await self._apply_results(result)
        return result

    async def _apply_results(self, result: dict) -> None:
        """Turn per-alpha winners into challenger parameter sets."""
        improvements: list[tuple[str, dict, float, dict]] = []
        for alpha, r in result.get("alphas", {}).items():
            best, base = r["best"], r["base"]
            if best.get("is_base"):
                continue
            if best["score"] <= -1e8:
                continue
            margin = 0.15 + 0.1 * best.get("distance", 0.0)
            base_score = base["score"] if base["score"] > -1e8 else -1.0
            if best["score"] > base_score + margin and best["test"].get("avg_r", 0) > 0 and best["test"].get("n", 0) >= 5:
                improvements.append((alpha, best["params"], best["score"] - base_score, best["test"]))
        improvements.sort(key=lambda x: -x[2])
        lines = []
        for alpha, params, gain, test in improvements[: self.s.challengers]:
            challenger = self.app.params.with_alpha(alpha, params, version=short_id("v"),
                                                    note=f"research:{alpha} (+{gain:.2f} score, OOS avgR {test.get('avg_r', 0):+.2f}, n={test.get('n', 0)})")
            slot = await self.app.install_challenger(challenger, source=alpha)
            if slot:
                lines.append(f"• {alpha}: 챌린저 {slot.name} 교체 (OOS n={test.get('n', 0)}, avgR {test.get('avg_r', 0):+.2f}, PF {test.get('profit_factor', 0):.2f})")
        summary = {a: {"best_score": round(r["best"]["score"], 2), "base_score": round(r["base"]["score"], 2),
                       "test_n": r["best"]["test"].get("n", 0), "test_avg_r": r["best"]["test"].get("avg_r", 0.0)}
                   for a, r in result.get("alphas", {}).items()}
        await self.app.emit("research_done", {"message": "\n".join(lines) if lines else "개선된 파라미터 없음 — 챔피언 유지",
                                              "summary": summary, "seconds": result.get("seconds"), "result": result})

    # --- promotion -----------------------------------------------------------------------------
    async def evaluate_challengers(self, force: bool = False) -> None:
        now = now_ms()
        if not force and now - self.last_promotion_check < 60 * 60_000:
            return
        self.last_promotion_check = now
        champ_engine = self.app.engines.get("paper")
        if champ_engine is None:
            return
        for slot in list(self.app.challengers):
            eng = self.app.engines.get(slot.name)
            if eng is None:
                continue
            ch_trades = self.store.load_trades(engine=slot.name, since=slot.started)
            champ_trades = self.store.load_trades(engine="paper", since=slot.started)
            if len(ch_trades) < self.s.promotion_min_trades or len(champ_trades) < max(10, self.s.promotion_min_trades // 2):
                # retire a challenger that is clearly bleeding before it reaches the trade count
                if len(ch_trades) >= 15 and summarize(ch_trades)["avg_r"] < -0.5:
                    await self.app.retire_challenger(slot, reason="조기 탈락 (avgR < -0.5)")
                continue
            st_ch, st_champ = summarize(ch_trades), summarize(champ_trades)
            # drawdown compared in R units: the two paper accounts have different equity bases
            st_ch["max_dd_r"], st_champ["max_dd_r"] = _max_dd_r(ch_trades), _max_dd_r(champ_trades)
            o_ch, o_champ = objective(st_ch, 10), objective(st_champ, 10)
            better = (o_ch > o_champ + 0.5 and st_ch["avg_r"] > st_champ["avg_r"] + 0.05
                      and st_ch["max_dd_r"] <= st_champ["max_dd_r"] * 1.2 + 1e-9 and st_ch["net"] > 0)
            if better:
                await self.app.promote(slot, st_ch, st_champ)
            elif len(ch_trades) >= self.s.promotion_min_trades * 2 and o_ch < o_champ:
                await self.app.retire_challenger(slot, reason="챔피언 대비 열위 (충분한 표본)")

    # --- graduation (paper -> live offer) -------------------------------------------------------
    def graduation_status(self) -> dict:
        since = now_ms() - 14 * MS_DAY
        trades = self.store.load_trades(engine="paper", since=since)
        st = summarize(trades)
        # load_equity keeps the NEWEST `limit` rows; the engine persists one row per minute, so a 14-day
        # window holds ~20k rows and limit=5000 would silently drop everything older than ~3.5 days
        # (x2 headroom for the extra rows written after restarts).
        curve = self.store.load_equity("paper", since=since, limit=2 * 14 * (MS_DAY // 60_000) + 10)
        from heartless.execution.stats import equity_drawdown

        _, dd_pct = equity_drawdown(curve)
        ok = (st["n"] >= self.s.graduation_min_trades and st["profit_factor"] >= self.s.graduation_min_profit_factor
              and dd_pct <= self.s.graduation_max_drawdown_pct and st["net"] > 0)
        return {"ok": ok, "n": st["n"], "profit_factor": st["profit_factor"], "max_dd_pct": dd_pct, "net": st["net"],
                "win_rate": st["win_rate"], "avg_r": st["avg_r"],
                "need": {"trades": self.s.graduation_min_trades, "profit_factor": self.s.graduation_min_profit_factor,
                         "max_dd_pct": self.s.graduation_max_drawdown_pct}}

    async def check_graduation(self) -> None:
        if self.app.mode == "live" or not self.s.live_capable:
            return
        day = now_ms() // MS_DAY
        if day == self.graduation_notified_day:
            return
        gs = self.graduation_status()
        if gs["ok"]:
            self.graduation_notified_day = day
            self.store.set("research.graduation_day", day)
            await self.app.emit("graduation_ready", gs)

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

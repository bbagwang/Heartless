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
                                                self.store.get("advisor.seeds", {}) or {}, self.s.research_folds)
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
        """Turn walk-forward results into challenger parameter sets.

        * better parameters for an enabled alpha -> challenger with those parameters;
        * a disabled alpha whose best candidate passes a stricter gate (positive in most confirmation folds) ->
          challenger with the alpha switched ON (self-healing: an alpha that starts working again comes back);
        * an enabled alpha that loses in every confirmation fold and has no better parameters -> challenger with
          the alpha switched OFF (self-pruning). Challengers still have to beat the champion in real-time paper."""
        proposals: list[tuple[float, str, StrategyParams, str]] = []
        for alpha, r in result.get("alphas", {}).items():
            best, base = r["best"], r["base"]
            enabled = r.get("enabled", True)
            k = int(r.get("n_folds", 1) or 1)
            need_folds = max(1, (2 * k + 2) // 3)  # 2 of 3, 3 of 4 ...
            base_score = base["score"] if base["score"] > -1e8 else -1.0
            test = best.get("test", {})
            if not best.get("is_base") and best["score"] > -1e8:
                margin = 0.15 + 0.1 * best.get("distance", 0.0)
                folds_ok = k <= 1 or best.get("folds_pos", 0) >= need_folds
                if best["score"] > base_score + margin and test.get("avg_r", 0) > 0 and test.get("n", 0) >= 5 and folds_ok:
                    if enabled:
                        ch = self.app.params.with_alpha(alpha, best["params"], version=short_id("v"),
                                                        note=f"research:{alpha} (+{best['score'] - base_score:.2f} score, "
                                                             f"OOS avgR {test.get('avg_r', 0):+.2f}, folds+ {best.get('folds_pos', 0)}/{k})")
                        proposals.append((best["score"] - base_score, alpha, ch,
                                          f"{alpha}: 파라미터 개선 (OOS n={test.get('n', 0)}, avgR {test.get('avg_r', 0):+.2f}, 구간+ {best.get('folds_pos', 0)}/{k})"))
                        continue
                    if test.get("avg_r", 0) >= 0.05 and test.get("n", 0) >= 15 and best.get("folds_pos", 0) >= need_folds:
                        ch = self.app.params.with_alpha(alpha, best["params"], version=short_id("v"),
                                                        note=f"research:enable {alpha} (OOS avgR {test.get('avg_r', 0):+.2f}, folds+ {best.get('folds_pos', 0)}/{k})")
                        ch.enabled[alpha] = True
                        proposals.append((best["score"] - base_score + 0.5, alpha, ch,
                                          f"{alpha}: 비활성 알파 재검증 통과 → 켠 챌린저 (OOS avgR {test.get('avg_r', 0):+.2f}, 구간+ {best.get('folds_pos', 0)}/{k})"))
                        continue
            if enabled and k > 1:
                folds = base.get("folds") or []
                losing = [f for f in folds if f.get("n", 0) >= 8 and f.get("avg_r", 0) < -0.1]
                if len(folds) == k and len(losing) == k:
                    ch = self.app.params.clone(version=short_id("v"), note=f"research:prune {alpha} (lost in all {k} folds)")
                    ch.enabled[alpha] = False
                    avg = sum(f.get("avg_r", 0) for f in folds) / k
                    proposals.append((0.25 + abs(avg), alpha, ch, f"{alpha}: 모든 검증 구간 손실 (평균 {avg:+.2f}R) → 끈 챌린저"))
        proposals.sort(key=lambda x: -x[0])
        lines = []
        for _, alpha, challenger, text in proposals[: self.s.challengers]:
            slot = await self.app.install_challenger(challenger, source=alpha)
            if slot:
                lines.append(f"• {text} → {slot.name}")
        summary = {a: {"best_score": round(r["best"]["score"], 2), "base_score": round(r["base"]["score"], 2),
                       "test_n": r["best"]["test"].get("n", 0), "test_avg_r": r["best"]["test"].get("avg_r", 0.0),
                       "folds_pos": r["best"].get("folds_pos", 0), "enabled": r.get("enabled", True)}
                   for a, r in result.get("alphas", {}).items()}
        await self.app.emit("research_done", {"message": "\n".join(lines) if lines else "개선된 파라미터 없음 — 챔피언 유지",
                                              "summary": summary, "seconds": result.get("seconds"), "result": result})

    # --- discovery ---------------------------------------------------------------------------------
    def discovery_due(self, now: int) -> bool:
        last = int(self.store.get("discovery.last_cycle", 0) or 0)
        return now - last >= self.s.discovery_interval_hours * 3_600_000

    async def discovery_cycle(self, force: bool = False) -> dict | None:
        """Mine rule-based alphas on the stored history and hand validated rule sets to a paper challenger."""
        if self.running:
            return None
        now = now_ms()
        if not force and not self.discovery_due(now):
            return None
        syms = [s for s in self.app.universe if self.store.candle_range(s)[0] is not None]
        ranges = [self.store.candle_range(s) for s in syms]
        if not ranges:
            return None
        first, last = max(r[0] for r in ranges), min(r[1] for r in ranges)
        days = (last - first) / MS_DAY
        self.store.set("discovery.last_cycle", now)
        if days < self.s.discovery_min_days:
            await self.app.emit("research_done", {"message": f"알파 발굴 건너뜀: 저장된 이력 {days:.0f}일 < {self.s.discovery_min_days}일", "notify": False})
            return None
        start = first + 14 * MS_DAY  # indicator warm-up
        split = start + int((last - start) * 0.7)
        self.running = True
        try:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            loop = asyncio.get_running_loop()
            await self.app.emit("research_started", {"message": f"알파 발굴 시작 ({len(syms)}종목, {days:.0f}일 이력)"})
            result = await loop.run_in_executor(self.pool, _discovery_worker, str(self.s.db_path), syms, (start, split - 1),
                                                (split, last))
        except Exception as e:  # noqa: BLE001
            log.exception("discovery cycle failed")
            if self.pool is not None:
                try:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001
                    pass
                self.pool = None
            self.running = False
            await self.app.emit("error", {"message": f"알파 발굴 실패: {e}"})
            return None
        self.running = False
        self.store.set("discovery.last", {"ts": now, **{k: result[k] for k in ("tested", "t_bar", "passed", "seconds", "rows")}})
        passed = result.get("passed") or []
        if passed:
            current = self.app.params.alphas.get("discovered", {}).get("rules") or []
            known = {r.get("id") for r in current}
            fresh = [r for r in passed if r.get("id") not in known]
            if fresh:
                rules = (fresh + current)[:6]
                ch = self.app.params.clone(version=short_id("v"), note=f"discovery: {len(fresh)} new rule(s)")
                ch.alphas.setdefault("discovered", {})["rules"] = rules
                ch.enabled["discovered"] = True
                slot = await self.app.install_challenger(ch, source="discovered")
                msg = f"알파 발굴: {result['tested']}개 규칙 검사, {len(fresh)}개 신규 규칙이 검증 통과 → 챌린저 {slot.name if slot else '-'}"
            else:
                msg = f"알파 발굴: 검증 통과 규칙이 이미 배포된 규칙과 동일 ({len(passed)}개)"
        else:
            msg = f"알파 발굴: {result['tested']}개 규칙 검사, 다중검정·검증 구간을 통과한 규칙 없음 (과최적화 차단)"
        await self.app.emit("research_done", {"message": msg, "seconds": result.get("seconds")})
        return result

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


def _discovery_worker(db_path: str, symbols: list[str], train: tuple[int, int], valid: tuple[int, int]) -> dict:
    """Runs in the research process: mine 15m and 1h rules, return the validated ones (JSON-serialisable)."""
    from heartless.learning import discovery as D

    out = {"tested": 0, "passed": [], "seconds": 0.0, "rows": {}, "t_bar": 0.0}
    for tf in ("15m", "1h"):
        try:
            res = D.mine(db_path, symbols, train, valid, tf, D.SearchConfig(beam=25, depth=3), workers=1)
        except Exception as e:  # noqa: BLE001
            out.setdefault("errors", []).append(f"{tf}: {e}")
            continue
        out["tested"] += res["tested"]
        out["passed"] += res["passed"]
        out["seconds"] += res["seconds"]
        out["rows"][tf] = res["rows"]
        out["t_bar"] = max(out["t_bar"], res["t_bar"])
    out["passed"].sort(key=lambda r: -r.get("stats", {}).get("valid", {}).get("t", 0.0))
    return out

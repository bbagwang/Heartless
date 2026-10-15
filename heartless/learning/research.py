"""Self-improvement loop: champion/challenger parameter management.

* Every `research_interval_minutes` a worker process runs walk-forward optimisation per alpha (the champion's enabled
  alphas, the disabled ones not marked failed and a rotation of the failed ones, ResearchManager.research_selection).
* Improvements become *challengers*: full parameter sets traded by dedicated paper engines on the
  live data feed, side by side with the champion.
* Challengers are judged on day-clustered statistics (trades clustered by UTC exit day, Newey-West across neighbouring
  days since E2). While the champion trades enough to compare with, a challenger must beat it on the paired daily-R
  difference (relative path); while it does
  not (e.g. no alpha enabled) a challenger must clear an absolute bar on its own record (absolute path), a stricter
  one when it switches on an alpha that failed out of sample (evidence registry). The live engine hot-swaps to a
  promoted champion. Retirement uses the same statistics.
* A per-alpha kill switch watches every engine's live (paper or real) closed trades and switches a losing alpha off
  in the champion at once (or retires a challenger); switching back on only happens through a promotion.
* In paper mode the champion's live-feed record is also checked against the graduation criteria,
  and the owner is offered a one-tap "go live" on Telegram.

The exact rules are in docs/SELF_IMPROVEMENT.md.
"""
from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing
import os
import statistics
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from heartless.execution.stats import HAC_LAG, cusum_lower, paired_daily_diff, r_stats, summarize
from heartless.learning import seal
from heartless.learning.optimizer import research_alphas, run_research_cycle
from heartless.strategy import evidence
from heartless.strategy.params import StrategyParams
from heartless.util.ids import short_id
from heartless.util.timeutil import MS_DAY, now_ms

log = logging.getLogger(__name__)

# --- promotion / retirement rules (ROADMAP P4, C2-B; docs/SELF_IMPROVEMENT.md 4절) ---------------------------------
# Promotion is decided at fixed checkpoints only (group-sequential design): PROMOTION_CHECKPOINTS calendar days after the
# challenger started (FAILED_CHECKPOINTS when it switches on an alpha that failed out of sample), each on its record up
# to that checkpoint. Re-checking a fixed bar every hour gave a challenger with no edge 17-19% to be promoted within 90
# days (4-7% at the failed-alpha bar; the relative bar was weaker still), because every look is another chance for
# noise. The t threshold of each checkpoint is calibrated by Monte Carlo (tools/calibrate_promotion.py,
# docs/results/c2b_promotion_calibration.json) so that a zero-edge challenger is promoted with probability <= 5% over
# its whole trial (<= 2% for the failed-alpha schedule) on either path, at 1-5 trades a day, fat-tailed R with win
# rates of 30-60% and same-day clustering. A challenger not promoted at its last checkpoint is retired (trial over).
# Retirement itself stays continuous (hourly): it is the safe direction.
# E2: both statistics are Newey-West t statistics over the calendar days (Bartlett kernel, PROMOTION_HAC_LAG days;
# heartless.execution.stats.day_hac_t / hac_t). The C2-B statistics assumed independent UTC days, and day factors that
# persist across days (AR(1), phi 0.3 / 0.6) or a zero-mean edge switching sign in 10-20 day blocks pushed the false
# promotion of the C2-B thresholds to 6-14% (failed-alpha bar 3.5-8%). The thresholds below are recalibrated over the
# C2-B scenarios plus those (docs/results/e2_promotion_calibration.json, docs/ROADMAP.md section 6).
PROMOTION_CHECKPOINTS = (14, 30, 60, 90)
FAILED_CHECKPOINTS = (30, 60, 90)
PROMOTION_HAC_LAG = HAC_LAG
# checkpoint (days) -> minimum t: absolute path (day-clustered HAC t, r_stats "t_hac") / relative path (paired
# daily-difference HAC t)
ABS_T_AT = {14: 6.88, 30: 4.7, 60: 3.32, 90: 2.71}
REL_T_AT = {14: 5.96, 30: 4.08, 60: 2.88, 90: 2.35}
FAILED_ABS_T_AT = {30: 5.69, 60: 4.02, 90: 3.28}
FAILED_REL_T_AT = {30: 4.92, 60: 3.48, 90: 2.84}
# Relative path (the champion closed >= PROMOTION_MIN_TRADES trades since the challenger started): the challenger needs
# PROMOTION_MIN_TRADES trades on >= PROMOTION_MIN_DAYS exit days, the checkpoint's paired t, a higher mean R and an R
# drawdown <= REL_DD_MULT x the champion's + REL_DD_SLACK_R.
REL_DD_MULT = 1.2
REL_DD_SLACK_R = 2.0
# Absolute path (the champion closed fewer trades than that, e.g. no alpha enabled): the challenger's own record must
# show max(PROMOTION_MIN_TRADES, ABS_MIN_TRADES) trades on >= PROMOTION_MIN_DAYS exit days, mean R > 0, the checkpoint's
# HAC t_day, PF (in R) >= ABS_MIN_PF_R and an R drawdown <= abs_dd_cap(n).
ABS_MIN_TRADES = 30
ABS_MIN_PF_R = 1.15
ABS_MAX_DD_R = 10.0
# ... and on either path, when it switches on (beyond the champion) an alpha the evidence registry marks 'failed' out
# of sample, the paper evidence has to be stronger: its schedule starts at FAILED_MIN_DAYS, it needs FAILED_MIN_DAYS
# exit days, and the thresholds are calibrated to 2%
FAILED_MIN_DAYS = 30
# Retirement: the early drop of a clearly bleeding challenger, else after 2x the path's minimum trades and days when
# the path's statistic (absolute: day-clustered HAC t; relative: paired-difference HAC t) is <= RETIRE_T
EARLY_DROP_TRADES = 15
EARLY_DROP_AVG_R = -0.5
RETIRE_T = -1.0


def abs_dd_cap(n: int) -> float:
    """Largest R drawdown the absolute path accepts: ABS_MAX_DD_R at ABS_MIN_TRADES trades, growing with sqrt(n) like
    the drawdown of a random walk. A flat 10R cap was sized for 30 trades / 14 days; over a 90-day trial a trend
    follower with a +0.20R edge at 2+ trades a day exceeded it in nearly every simulated run (median 24-40R), so the
    absolute path could never promote one (tools/calibrate_promotion.py, --flat-dd)."""
    return ABS_MAX_DD_R * math.sqrt(max(1.0, n / ABS_MIN_TRADES))


def checkpoints(failed: list[str] | tuple = ()) -> tuple[int, ...]:
    """The checkpoint schedule (days after the start) of a challenger; `failed` = the alphas it switches on beyond the
    champion that failed out of sample."""
    return FAILED_CHECKPOINTS if failed else PROMOTION_CHECKPOINTS


def promotion_bar(settings, path: str, failed: list[str] | tuple = (), checkpoint: int | None = None) -> dict:
    """The thresholds a challenger must meet on `path` ('relative' | 'absolute') at `checkpoint` (days after its start;
    default: the first of its schedule); `failed` = the alphas it enables beyond the champion that failed out of
    sample (the failed-alpha schedule: later start, FAILED_MIN_DAYS exit days, 2% thresholds)."""
    sched = checkpoints(failed)
    cp = checkpoint if checkpoint in sched else sched[0]
    min_days = max(FAILED_MIN_DAYS, settings.promotion_min_days) if failed else settings.promotion_min_days
    if path == "relative":
        bar = {"min_trades": settings.promotion_min_trades, "min_days": min_days,
               "min_paired_t": (FAILED_REL_T_AT if failed else REL_T_AT)[cp], "dd_mult": REL_DD_MULT,
               "dd_slack_r": REL_DD_SLACK_R}
    else:
        t_min = (FAILED_ABS_T_AT if failed else ABS_T_AT)[cp]
        # min_t_hac is the bar of r_stats "t_hac"; min_t_day carries the same value for renderers written before E2
        bar = {"min_trades": max(settings.promotion_min_trades, ABS_MIN_TRADES), "min_days": min_days, "min_avg_r": 0.0,
               "min_t_hac": t_min, "min_t_day": t_min, "min_pf_r": ABS_MIN_PF_R, "max_dd_r": ABS_MAX_DD_R}
    bar.update(checkpoint=cp, checkpoints=list(sched), hac_lag=PROMOTION_HAC_LAG)
    if failed:
        bar["failed_alphas"] = list(failed)
    return bar


def _round(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = 99.0 if v == float("inf") else round(v, 4)
        else:
            out[k] = v
    return out


def _before(trades: list[dict], cutoff: int) -> list[dict]:
    return [t for t in trades if (t.get("exit_time") or 0) < cutoff]


def judge_challenger(ch_trades: list[dict], champ_trades: list[dict], settings, start_ms: int, now: int,
                     new_alphas: list[str] | tuple = ()) -> dict:
    """Promotion / retirement verdict for one challenger at `now` (both records: trades opened since `start_ms`).

    Retirement is judged on the whole record so far (continuous). Promotion is judged only at the latest checkpoint
    passed (`checkpoint`, days after the start), on both records cut at that checkpoint, with that checkpoint's
    thresholds: re-evaluating it every hour until the next checkpoint gives the same verdict, so the number of looks
    is the number of checkpoints. Past the last checkpoint an unpromoted challenger is retired.

    The path statistics are Newey-West t statistics over the calendar days (lag PROMOTION_HAC_LAG): the record's
    `t_hac` on the absolute path, the paired daily-difference `paired.t` on the relative one (`t_day` and
    `paired.t_iid` are the lag-0 values, shown for comparison).

    Returns the decision with every input, threshold and check, so it can be stored and shown as is: `verdict` is
    'promote' | 'retire' | 'hold', `path` 'relative' | 'absolute' (docs/SELF_IMPROVEMENT.md)."""
    failed = evidence.failed(new_alphas)
    sched = checkpoints(failed)
    passed = [c for c in sched if now >= start_ms + c * MS_DAY]
    cp = passed[-1] if passed else None
    upcoming = [c for c in sched if c not in passed]
    # retirement: the whole record so far, on the path the champion's record selects now
    lag = PROMOTION_HAC_LAG
    cur, champ_cur = r_stats(ch_trades, lag), r_stats(champ_trades, lag)
    cur_path = "relative" if champ_cur["n"] >= settings.promotion_min_trades else "absolute"
    cur_bar = promotion_bar(settings, cur_path, failed, cp)
    if cur_path == "relative":
        retire_stat = paired_daily_diff(ch_trades, champ_trades, start_ms, now, lag=lag)["t"]
    else:
        retire_stat = cur["t_hac"]
    # promotion: the records cut at the checkpoint (before the first one: the record so far, for display)
    cutoff = start_ms + cp * MS_DAY if cp is not None else int(now) + 1
    ch_t, champ_t = (_before(ch_trades, cutoff), _before(champ_trades, cutoff)) if cp is not None else \
        (ch_trades, champ_trades)
    ch, champ = (r_stats(ch_t, lag), r_stats(champ_t, lag)) if cp is not None else (cur, champ_cur)
    path = "relative" if champ["n"] >= settings.promotion_min_trades else "absolute"
    bar = promotion_bar(settings, path, failed, cp)
    d: dict = {"path": path, "since": int(start_ms), "at": int(now), "checkpoint": cp,
               "next_checkpoint": upcoming[0] if upcoming else None, "checkpoints": list(sched),
               "cutoff": int(cutoff) if cp is not None else None, "challenger": _round(ch), "champion": _round(champ),
               "current": _round(cur), "new_alphas": list(new_alphas), "failed_alphas": failed, "bar": bar,
               "stat": {"kind": "newey-west", "kernel": "bartlett", "lag_days": lag}}
    checks: dict[str, bool] = {f"n >= {bar['min_trades']}": ch["n"] >= bar["min_trades"],
                               f"days >= {bar['min_days']}": ch["days"] >= bar["min_days"]}
    if path == "relative":
        pdiff = paired_daily_diff(ch_t, champ_t, start_ms, cutoff - 1, lag=lag)
        d["paired"] = _round({"days": pdiff["days"], "mean": pdiff["mean"], "t": pdiff["t"], "t_iid": pdiff["t_iid"]})
        dd_cap = REL_DD_MULT * champ["max_dd_r"] + REL_DD_SLACK_R
        checks[f"paired t >= {bar['min_paired_t']:g}"] = pdiff["t"] >= bar["min_paired_t"]
        checks["avgR > champion"] = ch["avg_r"] > champ["avg_r"]
        checks[f"DD <= {dd_cap:.2f}R"] = ch["max_dd_r"] <= dd_cap + 1e-9
    else:
        dd_cap = abs_dd_cap(ch["n"])
        checks["avgR > 0"] = ch["avg_r"] > 0
        checks[f"HAC t_day >= {bar['min_t_hac']:g}"] = ch["t_hac"] >= bar["min_t_hac"]
        checks[f"PF >= {ABS_MIN_PF_R:g}"] = ch["pf_r"] >= ABS_MIN_PF_R
        checks[f"DD <= {dd_cap:.2f}R"] = ch["max_dd_r"] <= dd_cap + 1e-9
    d["checks"] = checks
    d["retire_stat"] = round(retire_stat, 4)
    d["retire_path"] = cur_path
    if cur["n"] >= EARLY_DROP_TRADES and cur["avg_r"] < EARLY_DROP_AVG_R:
        d["verdict"], d["rule"] = "retire", f"early drop: avgR < {EARLY_DROP_AVG_R:g} after >= {EARLY_DROP_TRADES} trades"
    elif (cur["n"] >= 2 * cur_bar["min_trades"] and cur["days"] >= 2 * cur_bar["min_days"] and retire_stat <= RETIRE_T):
        what = "HAC paired t" if cur_path == "relative" else "HAC t_day"
        d["verdict"], d["rule"] = "retire", f"{what} <= {RETIRE_T:g} after 2x the minimum trades and days"
    elif cp is not None and all(checks.values()):
        d["verdict"], d["rule"] = "promote", f"{path} bar met at the {cp}-day checkpoint"
    elif cp is not None and not upcoming:
        d["verdict"], d["rule"] = "retire", f"trial over: not promoted at the last checkpoint ({cp} days)"
    elif cp is None:
        d["verdict"], d["rule"] = "hold", f"before the first checkpoint ({sched[0]} days)"
    else:
        d["verdict"], d["rule"] = "hold", f"bar not met at the {cp}-day checkpoint (next: {upcoming[0]} days)"
    return d


def describe_selection(sel: dict) -> str:
    """Korean one-liner of a research cycle's alpha selection (ResearchManager.research_selection)."""
    parts = []
    if sel.get("enabled"):
        parts.append("켜진 알파 " + "·".join(sel["enabled"]))
    if sel.get("untested"):
        parts.append("꺼진 미탈락 알파 " + "·".join(sel["untested"]))
    if sel.get("failed"):
        parts.append(f"탈락 알파 순환 {len(sel['failed'])}/{sel.get('failed_total', len(sel['failed']))} "
                     + "·".join(sel["failed"]))
    return ", ".join(parts) or "없음"


def describe_decision(d: dict) -> str:
    """Korean one-to-four-line rendering of a decision's inputs for Telegram, logs and version notes. Since E2 the path
    statistics are HAC t values (Newey-West over `bar.hac_lag` days); a decision stored before E2 has none and shows
    its lag-0 t_day / paired t as it was judged."""
    ch, champ, bar = d.get("challenger", {}), d.get("champion", {}), d.get("bar", {})
    cp, nxt = d.get("checkpoint"), d.get("next_checkpoint")
    sched = "·".join(str(c) for c in d.get("checkpoints") or bar.get("checkpoints") or [])
    when = (f"{cp}일 판정점 기록" if cp is not None else "첫 판정점 전 현재 기록") + \
        (f" · 다음 판정점 {nxt}일" if nxt else " · 마지막 판정점") + (f" (판정점 {sched}일)" if sched else "")
    hac = f"HAC({bar['hac_lag']}일) " if bar.get("hac_lag") else ""
    t_rec = (f"{hac}t_day {ch['t_hac']:+.2f} (일 독립 가정 {ch.get('t_day', 0):+.2f})" if "t_hac" in ch
             else f"t_day {ch.get('t_day', 0):+.2f}")
    rec = (f"챌린저 n={ch.get('n', 0)} (청산일 {ch.get('days', 0)}일), 평균 {ch.get('avg_r', 0):+.3f}R, "
           f"{t_rec}, PF {min(ch.get('pf_r', 0), 99):.2f}, 낙폭 {ch.get('max_dd_r', 0):.1f}R")
    if d.get("path") == "relative":
        pr = d.get("paired", {})
        head = (f"상대 경로: {hac}일별 차이 t {pr.get('t', 0):+.2f} ({pr.get('days', 0)}일, 평균 {pr.get('mean', 0):+.3f}R/일) · "
                f"챔피언 n={champ.get('n', 0)} 평균 {champ.get('avg_r', 0):+.3f}R 낙폭 {champ.get('max_dd_r', 0):.1f}R")
        need = (f"기준 n≥{bar.get('min_trades')}·청산일 {bar.get('min_days')}일, {hac}차이 t≥{bar.get('min_paired_t', 0):g}, "
                f"평균R>챔피언, 낙폭≤{bar.get('dd_mult', 0):g}×챔피언+{bar.get('dd_slack_r', 0):g}R")
    else:
        head = f"절대 경로 (챔피언 거래 {champ.get('n', 0)}건 < 비교 최소치)"
        t_min = bar.get("min_t_hac", bar.get("min_t_day", 0))
        need = (f"기준 n≥{bar.get('min_trades')}·청산일 {bar.get('min_days')}일, 평균R>0, {hac}t_day≥{t_min:g}, "
                f"PF≥{bar.get('min_pf_r', 0):g}, 낙폭≤{abs_dd_cap(int(ch.get('n', 0) or 0)):.1f}R")
    if d.get("failed_alphas"):
        need += f" (OOS 탈락 알파 {', '.join(d['failed_alphas'])}: 강화 기준)"
    cur = d.get("current")
    if cur and cp is not None and cur.get("n") != ch.get("n"):
        when += f" · 현재 n={cur.get('n', 0)} 평균 {cur.get('avg_r', 0):+.3f}R"
    return f"{head}\n{when}\n{rec}\n{need}"


# --- challenger base and diff (C2-B) ------------------------------------------------------------------------------------
# A challenger is a CHANGE to the champion it was created from (its base): the alpha values, switches and ensemble keys
# it changes. A promotion applies only that change onto the champion of the moment, so a promotion or a kill-switch
# trip that happened in between is kept; the change is refused (the challenger retired, with the reason logged) when it
# would undo one of them: re-enabling an alpha the kill switch switched off after the base, or touching an alpha /
# ensemble key that a later champion change set differently from the base.
def params_diff(base: StrategyParams, params: StrategyParams) -> tuple[dict, dict]:
    """(diff, base_values): what `params` changes relative to `base`, and the base's values of those keys."""
    diff: dict = {"alphas": {}, "enabled": {}, "ensemble": {}}
    old: dict = {"alphas": {}, "enabled": {}, "ensemble": {}}
    for a, vals in params.alphas.items():
        bv = base.alphas.get(a, {})
        changed = {k: v for k, v in vals.items() if bv.get(k) != v}
        if changed:
            diff["alphas"][a] = changed
            old["alphas"][a] = {k: bv.get(k) for k in changed}
    for a, on in params.enabled.items():
        if bool(base.enabled.get(a, False)) != bool(on):
            diff["enabled"][a] = bool(on)
            old["enabled"][a] = bool(base.enabled.get(a, False))
    for k, v in params.ensemble.items():
        if base.ensemble.get(k) != v:
            diff["ensemble"][k] = v
            old["ensemble"][k] = base.ensemble.get(k)
    return diff, old


def diff_alphas(diff: dict) -> list[str]:
    """The alphas a diff touches (values or switch)."""
    return sorted(set(diff.get("alphas") or {}) | set(diff.get("enabled") or {}))


def apply_diff(champion: StrategyParams, diff: dict, version: str | None = None, note: str = "") -> StrategyParams:
    """The champion with `diff` applied (the order constraints of an alpha's values are enforced as in with_alpha)."""
    new = champion.clone(version=version, note=note)
    for a, vals in (diff.get("alphas") or {}).items():
        if a in new.alphas:
            new = new.with_alpha(a, dict(vals), version=new.version, note=note)
    for a, on in (diff.get("enabled") or {}).items():
        if a in new.alphas:
            new.enabled[a] = bool(on)
    for k, v in (diff.get("ensemble") or {}).items():
        new.ensemble[k] = v
    return new


def diff_conflicts(diff: dict, base_values: dict, base: dict | None, current: StrategyParams) -> list[str]:
    """Keys of `diff` whose current champion value is neither the base's value nor the diff's own: a later champion
    change set them differently. For every alpha the diff touches, its switch and values in the current champion are
    compared with the base's (`base`: the base champion's params dict when stored, else only `base_values`)."""
    out: list[str] = []
    for a in diff_alphas(diff):
        b_on = (base_values.get("enabled") or {}).get(a, (base or {}).get("enabled", {}).get(a))
        if b_on is not None:
            c_on, target = bool(current.enabled.get(a, False)), bool((diff.get("enabled") or {}).get(a, b_on))
            if c_on != bool(b_on) and c_on != target:
                out.append(f"{a}: switched {'on' if c_on else 'off'} since the base")
        b_vals = dict((base or {}).get("alphas", {}).get(a, {}))
        b_vals.update((base_values.get("alphas") or {}).get(a, {}))
        for k, bv in b_vals.items():
            cv = current.alphas.get(a, {}).get(k)
            if cv != bv and cv != (diff.get("alphas") or {}).get(a, {}).get(k, bv):
                out.append(f"{a}.{k}: {bv!r} -> {cv!r} since the base")
    for k, bv in (base_values.get("ensemble") or {}).items():
        cv = current.ensemble.get(k)
        if cv != bv and cv != (diff.get("ensemble") or {}).get(k):
            out.append(f"ensemble.{k}: {bv!r} -> {cv!r} since the base")
    return out


@dataclass
class ChallengerSlot:
    name: str
    params: StrategyParams
    started: int
    source: str = ""
    # the champion version it was created from (its base), when that was created, and what it changes relative to it
    # (params_diff): a promotion applies only `diff` onto the champion of the moment. Empty for a slot built without
    # them (older stored challengers, tests): ResearchManager.slot_diff then takes the current champion as the base.
    base: str = ""
    base_created: int = 0
    diff: dict = field(default_factory=dict)
    base_values: dict = field(default_factory=dict)
    base_params: dict | None = None


class ResearchManager:
    def __init__(self, app):
        self.app = app
        self.s = app.s
        self.store = app.store
        self.pool: ProcessPoolExecutor | None = None
        self.last_cycle_ts: int = int(self.store.get("research.last_cycle", 0) or 0)
        # wall-clock length of the last finished cycle (ms): due() keeps the worker's research duty cycle <= 1/2
        self.last_cycle_ms: int = int(self.store.get("research.last_cycle_ms", 0) or 0)
        self.last_summary: dict = self.store.get("research.last_summary", {}) or {}
        self.last_promotion_check: int = 0
        self.graduation_notified_day: int = int(self.store.get("research.graduation_day", 0) or 0)
        self.running = False
        self.history: list[dict] = []
        self.research_failures: int = 0
        self.last_error: str = ""
        # slot name -> the latest promotion / retirement decision (inputs, thresholds, verdict) for /challengers
        self.last_decisions: dict[str, dict] = {}
        self.stopped = False  # set by shutdown(): a cycle whose worker was terminated ends quietly

    # --- champion persistence ------------------------------------------------------------------
    def load_champion(self) -> StrategyParams:
        rows = self.store.load_params_versions(role="champion", limit=1)
        if rows:
            p, reset = StrategyParams.from_stored(rows[0]["params"])
            p.version = rows[0]["id"]
            p.note = rows[0].get("note", "")
            if reset:
                # the stored champion was tuned / switched on for alpha designs that have since been replaced or
                # re-validated: keep its other settings, reset those alphas to the shipped defaults, persist as new version
                new = p.clone(note=f"design change: {', '.join(reset)} reset to shipped defaults (was {p.version})")
                log.warning("stored champion %s predates the current design of %s; using shipped defaults for them as "
                            "champion %s", p.version, ", ".join(reset), new.version)
                self.store.set_params_role(p.version, "retired")
                self.store.save_params_version(new.version, new.created, "champion", "design-reset", new.note,
                                               new.to_dict())
                return new
            return p
        p = StrategyParams.default()
        self.store.save_params_version(p.version, p.created, "champion", "default", p.note, p.to_dict())
        return p

    def load_challengers(self) -> list[ChallengerSlot]:
        slots: list[ChallengerSlot] = []
        for row in self.store.load_params_versions(role="challenger", limit=self.s.challengers):
            p, reset = StrategyParams.from_stored(row["params"])
            if reset:  # its experiment was about an alpha design that no longer exists
                log.warning("retiring challenger %s: stored for an older design of %s", row["id"], ", ".join(reset))
                self.store.set_params_role(row["id"], "retired")
                continue
            p.version = row["id"]
            m = row["metrics"]
            slot = ChallengerSlot(name=m.get("slot", f"challenger-{len(slots) + 1}"), params=p,
                                  started=m.get("started", row["created"]), source=row.get("note", ""),
                                  base=m.get("base", ""), base_created=int(m.get("base_created", 0) or 0),
                                  diff=m.get("diff") or {}, base_values=m.get("base_values") or {},
                                  base_params=m.get("base_params"))
            if not m.get("diff"):
                # an older stored challenger without a recorded base: the current champion, recorded now. Derived
                # again at every start, it would be a later champion after a promotion or a kill-switch trip, and
                # the challenger's stale values of what that champion change set would join its diff (a promotion
                # would then undo the change)
                self.slot_diff(slot)
                if slot.diff:
                    self.store.update_params_metrics(row["id"], {
                        "base": slot.base, "base_created": slot.base_created, "diff": slot.diff,
                        "base_values": slot.base_values, "base_params": slot.base_params})
            slots.append(slot)
        return slots

    # --- research cycle ------------------------------------------------------------------------
    def worker_settings(self) -> dict:
        """The settings handed to the research worker (no credentials). CONTEXT_DB names the bot's live context store
        (app.live_context_path: the hourly snapshots it published plus the hours built from stored candles), so the
        walk-forward backtests (load_backtester(market="auto")) see the market context the alphas see live."""
        d = {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in self.s.model_dump(by_alias=True).items()
             if k not in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY")}
        ctx = getattr(self.app, "live_context_path", None)
        if ctx:
            d["CONTEXT_DB"] = str(ctx)
        return d

    def due(self, now: int) -> bool:
        """RESEARCH_INTERVAL_MINUTES after the last cycle started, and at least twice the last cycle's length (E2): the
        worker spends at most half of its time on research even when the champion enables more alphas than the
        default interval was measured with, which leaves the daily discovery (run only while the worker is idle)
        its turn."""
        return (now - self.last_cycle_ts) >= self.interval_ms()

    def interval_ms(self) -> int:
        """The gap due() waits after a cycle's start: RESEARCH_INTERVAL_MINUTES, or twice the last cycle's length."""
        return max(self.s.research_interval_minutes * 60_000, 2 * self.last_cycle_ms)

    def next_due_ts(self) -> int | None:
        """When the next cycle becomes due (status UIs); None before the first cycle, which is due at once."""
        return self.last_cycle_ts + self.interval_ms() if self.last_cycle_ts else None

    # --- which alphas a cycle tunes (E2) ---------------------------------------------------------
    ROTATION_KEY = "research.failed_rotation"

    def research_selection(self, advance: bool = False) -> dict:
        """The alphas the next research cycle tunes: (a) the champion's enabled alphas, (b) its disabled alphas that the
        evidence registry does not mark 'failed' (untested, passed), (c) RESEARCH_FAILED_PER_CYCLE of the failed
        disabled ones, in a fixed rotation by name that continues after the last one taken (kv ROTATION_KEY, so a
        restart resumes it), which revisits each failed alpha every ceil(failed / k) cycles. Tuning every alpha every
        cycle spent most of the worker's time on alphas that already failed out of sample. `advance` records the
        rotation's progress (at submission, so an alpha that crashes the worker does not block the rotation)."""
        champ = self.app.params
        tunable = research_alphas(champ)
        enabled = [a for a in tunable if champ.enabled.get(a, False)]
        off = [a for a in tunable if not champ.enabled.get(a, False)]
        failed = sorted(a for a in off if evidence.verdict(a) == "failed")
        fresh = [a for a in off if a not in failed]
        k = max(0, min(int(self.s.research_failed_per_cycle), len(failed)))
        last = str(self.store.get(self.ROTATION_KEY, "") or "")
        start = next((i for i, a in enumerate(failed) if a > last), 0)
        rotated = [failed[(start + i) % len(failed)] for i in range(k)]
        if advance and rotated:
            self.store.set(self.ROTATION_KEY, rotated[-1])
        return {"enabled": enabled, "untested": fresh, "failed": rotated, "failed_total": len(failed),
                "alphas": enabled + fresh + rotated}

    async def _await_context_fill(self) -> None:
        """Wait for the app's start-up fill of the live context store (app._fill_context_gaps). The worker loads the
        context of its window once, when its backtester is built, and a cycle is due at once after a start that
        followed a downtime of more than RESEARCH_INTERVAL_MINUTES (or on a fresh install): submitted while the fill
        still runs, it backtested the hours not yet built with an empty ctx.market."""
        task = getattr(self.app, "_ctx_fill_task", None)
        if task is not None and not task.done():
            log.info("research cycle waits for the live context store's start-up fill")
            await asyncio.wait([task])  # never raises the fill's own error or cancellation (it logs its failures)

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
            await self._await_context_fill()
            if self.stopped:
                self.running = False
                return None
            if self.pool is None:
                self.pool = _new_pool()
            loop = asyncio.get_running_loop()
            settings_dict = self.worker_settings()
            symbols_dict = {k: asdict(v) for k, v in self.app.symbols.items() if k in self.app.universe}
            sel = self.research_selection(advance=True)
            if not sel["alphas"]:
                self.running = False
                await self.app.emit("research_done", {"message": "리서치 건너뜀: 조정할 알파 없음", "notify": False})
                return None
            await self.app.emit("research_started", {"message": "퀀트 리서치 사이클 시작 (워크포워드 최적화: "
                                                                + describe_selection(sel) + ")"})
            t_wall = time.monotonic()
            result = await loop.run_in_executor(self.pool, run_research_cycle, str(self.s.db_path), settings_dict,
                                                self.app.params.to_dict(), symbols_dict, self.s.research_lookback_days,
                                                self.s.research_candidates, now % 100_000, sel["alphas"],
                                                self.store.get("advisor.seeds", {}) or {}, self.s.research_folds)
            self.last_cycle_ms = int((time.monotonic() - t_wall) * 1000)
            self.store.set("research.last_cycle_ms", self.last_cycle_ms)
        except Exception as e:  # noqa: BLE001
            if self.stopped:  # the worker was terminated by shutdown(): not a failure
                self.running = False
                return None
            log.exception("research cycle failed")
            # A crashed worker (OOM kill, os._exit) leaves the executor permanently broken: every later
            # submit() raises BrokenProcessPool immediately. Discard it so the next cycle recreates one.
            self._discard_pool()
            self.research_failures += 1
            self.last_error = f"{now}: {e}"
            await self.app.emit("error", {"message": f"리서치 사이클 실패: {e}"})
            self.running = False
            return None
        self.running = False
        if "error" in result:
            await self.app.emit("research_done", {"message": f"리서치 건너뜀: {result['error']}", "result": result})
            return result
        result["selection_detail"] = sel
        self.last_summary = result
        self.store.set("research.last_summary", result)
        self.store.save_research_run(now, "all", result)
        await self._apply_results(result)
        return result

    async def _apply_results(self, result: dict) -> None:
        """Turn walk-forward results into challenger parameter sets.

        * better parameters for an enabled alpha -> challenger with those parameters;
        * a disabled alpha whose best candidate passes a stricter gate (positive in most confirmation folds) ->
          challenger with the alpha switched ON (self-healing: an alpha that starts working again comes back); this
          includes the case where the best candidate is the alpha's current parameters;
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
            if (not enabled and best.get("is_base") and best["score"] > -1e8 and test.get("avg_r", 0) >= 0.05
                    and test.get("n", 0) >= 15 and best.get("folds_pos", 0) >= need_folds):
                # a disabled alpha whose CURRENT parameters now pass the stricter gate (no better candidate needed)
                ch = self.app.params.clone(version=short_id("v"),
                                           note=f"research:enable {alpha} as is (OOS avgR {test.get('avg_r', 0):+.2f}, folds+ {best.get('folds_pos', 0)}/{k})")
                ch.enabled[alpha] = True
                proposals.append((0.5, alpha, ch, f"{alpha}: 비활성 알파가 현재 파라미터로 재검증 통과 → 켠 챌린저 "
                                                  f"(OOS avgR {test.get('avg_r', 0):+.2f}, 구간+ {best.get('folds_pos', 0)}/{k})"))
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
                lines.append(f"• {text} → {slot.name} ({self.bar_note(slot)})")
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
        win = seal.clip_window(start, last)  # the automated loop never mines or validates on a sealed period
        if win is None or win[1] - win[0] < (self.s.discovery_min_days - 14) * MS_DAY:
            await self.app.emit("research_done", {"message": "알파 발굴 건너뜀: 봉인 구간을 제외하면 이력이 부족함", "notify": False})
            return None
        start, last = win
        split = start + int((last - start) * 0.7)
        self.running = True
        try:
            if self.pool is None:
                self.pool = _new_pool()
            loop = asyncio.get_running_loop()
            await self.app.emit("research_started", {"message": f"알파 발굴 시작 ({len(syms)}종목, {days:.0f}일 이력)"})
            result = await loop.run_in_executor(self.pool, _discovery_worker, str(self.s.db_path), syms, (start, split - 1),
                                                (split, last), self.s.discovery_grids)
        except Exception as e:  # noqa: BLE001
            if self.stopped:
                self.running = False
                return None
            log.exception("discovery cycle failed")
            self._discard_pool()
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
                msg = (f"알파 발굴: {result['tested']}개 규칙 검사, {len(fresh)}개 신규 규칙이 검증 통과 → 챌린저 "
                       f"{slot.name if slot else '-'}" + (f" ({self.bar_note(slot)})" if slot else ""))
            else:
                msg = f"알파 발굴: 검증 통과 규칙이 이미 배포된 규칙과 동일 ({len(passed)}개)"
        else:
            msg = f"알파 발굴: {result['tested']}개 규칙 검사, 다중검정·검증 구간을 통과한 규칙 없음 (과최적화 차단)"
        await self.app.emit("research_done", {"message": msg, "seconds": result.get("seconds")})
        return result

    # --- promotion -----------------------------------------------------------------------------
    def new_alphas(self, params: StrategyParams) -> list[str]:
        """Alphas `params` switches on that the current champion has off (what a promotion would switch on)."""
        champ = self.app.params.enabled
        return [a for a, on in params.enabled.items() if on and not champ.get(a, False) and a in params.alphas]

    def slot_diff(self, slot: ChallengerSlot) -> dict:
        """The challenger's change relative to its base (filled from the current champion when it was not recorded)."""
        if not slot.diff:
            champ = getattr(self.app, "params", None)
            if champ is None:
                return {"alphas": {}, "enabled": {}, "ensemble": {}}
            slot.diff, slot.base_values = params_diff(champ, slot.params)
            slot.base, slot.base_created = champ.version, int(slot.base_created or slot.started or 0)
            slot.base_params = champ.to_dict()
        return slot.diff

    def slot_new_alphas(self, slot: ChallengerSlot) -> list[str]:
        """Alphas the challenger's promotion would switch on: those its diff switches on that the champion has off."""
        champ = self.app.params
        return [a for a, on in (self.slot_diff(slot).get("enabled") or {}).items()
                if on and not champ.enabled.get(a, False) and a in champ.alphas]

    def killed_since(self, alpha: str, since: int) -> dict | None:
        """The newest kill-switch champion version created after `since` that switched `alpha` off, or None."""
        for row in self.store.params_versions_since(int(since), source="kill-switch"):
            decisions = (row.get("metrics") or {}).get("decisions") or [{}]
            if any(d.get("alpha") == alpha for d in decisions):
                return row
        return None

    def promotion_blockers(self, slot: ChallengerSlot) -> list[str]:
        """Why the challenger's change must not be applied onto the current champion (empty: it may): it re-enables an
        alpha the kill switch switched off after the challenger's base, or a later champion change set a key it touches
        differently (diff_conflicts)."""
        diff = self.slot_diff(slot)
        out = []
        for a, on in sorted((diff.get("enabled") or {}).items()):
            row = self.killed_since(a, slot.base_created or slot.started) if on else None
            if row is not None:
                out.append(f"kill switch switched {a} off in {row['id']} after the challenger's base {slot.base or '-'}")
        return out + diff_conflicts(diff, slot.base_values, slot.base_params, self.app.params)

    def bar_note(self, slot: ChallengerSlot) -> str:
        """Short Korean note of the promotion bar a freshly installed challenger faces (shown when it is installed).
        The path is decided at each checkpoint by the champion's trades over the comparison period; its trade count
        over the last PROMOTION_MIN_DAYS days is the forecast."""
        champ_n = len(self.store.load_trades(engine="paper", since=now_ms() - self.s.promotion_min_days * MS_DAY))
        failed = evidence.failed(self.slot_new_alphas(slot))
        sched = checkpoints(failed)
        days = "·".join(str(c) for c in sched)
        if champ_n >= self.s.promotion_min_trades:
            ts = "/".join(f"{(FAILED_REL_T_AT if failed else REL_T_AT)[c]:g}" for c in sched)
            bar = promotion_bar(self.s, "relative", failed)
            note = f"상대 경로: 판정점 {days}일에 챔피언 대비 일별 차이 HAC t≥{ts}, 청산일 {bar['min_days']}일 이상"
        else:
            ts = "/".join(f"{(FAILED_ABS_T_AT if failed else ABS_T_AT)[c]:g}" for c in sched)
            bar = promotion_bar(self.s, "absolute", failed)
            note = (f"승격 기준: 절대 경로 판정점 {days}일에 HAC t_day≥{ts}, n≥{bar['min_trades']}·청산일 "
                    f"{bar['min_days']}일, PF≥{bar['min_pf_r']:g}, 낙폭≤{bar['max_dd_r']:g}R×√(n/{ABS_MIN_TRADES})")
        return note + (f"; OOS 탈락 알파 {', '.join(failed)} 강화 기준" if failed else "") + \
            f"; {sched[-1]}일까지 승격되지 않으면 종료"

    def trial_days(self, slot: ChallengerSlot) -> int:
        """Length of the challenger's trial in days: its last checkpoint (FAILED_CHECKPOINTS when it switches on an
        alpha that failed out of sample). Its slot is protected from replacement until then unless it is losing on a
        promotion-sized sample (app.install_challenger): most of the power to detect a real edge comes from the late
        checkpoints, so evicting a challenger after the first one would leave almost nothing promotable."""
        return checkpoints(evidence.failed(self.slot_new_alphas(slot)))[-1]

    async def evaluate_challengers(self, force: bool = False) -> None:
        """Hourly: judge every challenger on its record since it started (judge_challenger: retirement continuously,
        promotion at its checkpoints) and act on the verdict. Every decision is logged; a promotion or retirement
        stores its inputs with the parameter versions involved and shows them in the Telegram message. A promotion
        whose change can no longer be applied onto the current champion (promotion_blockers) retires the challenger
        instead, with the reason."""
        now = now_ms()
        if not force and now - self.last_promotion_check < 60 * 60_000:
            return
        self.last_promotion_check = now
        if self.app.engines.get("paper") is None:
            return
        for slot in list(self.app.challengers):
            if self.app.engines.get(slot.name) is None:
                continue
            # both records over the same period: trades OPENED since the challenger started (a champion position opened
            # before it existed and closed after is not something the challenger could have taken)
            ch_trades = [t for t in self.store.load_trades(engine=slot.name, since=slot.started)
                         if (t.get("entry_time") or 0) >= slot.started]
            champ_trades = [t for t in self.store.load_trades(engine="paper", since=slot.started)
                            if (t.get("entry_time") or 0) >= slot.started]
            d = judge_challenger(ch_trades, champ_trades, self.s, slot.started, now, self.slot_new_alphas(slot))
            d.update(slot=slot.name, version=slot.params.version, source=slot.source,
                     champion_version=self.app.params.version, base=slot.base, diff=self.slot_diff(slot))
            if d["verdict"] == "promote":
                blockers = self.promotion_blockers(slot)
                if blockers:
                    d.update(verdict="retire", rule="change conflicts with the current champion: " + "; ".join(blockers),
                             blockers=blockers)
            self.last_decisions[slot.name] = d
            log.info("challenger %s (%s) %s path, checkpoint %s: %s (%s) n=%d days=%d avgR %+.3f HAC t_day %+.2f "
                     "(lag-0 %+.2f)%s", slot.name, slot.params.version, d["path"], d["checkpoint"], d["verdict"],
                     d["rule"], d["challenger"]["n"], d["challenger"]["days"], d["challenger"]["avg_r"],
                     d["challenger"]["t_hac"], d["challenger"]["t_day"],
                     f" HAC paired t {d['paired']['t']:+.2f}" if "paired" in d else "")
            if d["verdict"] == "promote":
                await self.app.promote(slot, d)
                break  # the other challengers are judged against the new champion at the next check
            if d["verdict"] == "retire":
                if d.get("blockers"):
                    log.warning("challenger %s (%s) met the bar but is not promoted: %s", slot.name,
                                slot.params.version, "; ".join(d["blockers"]))
                    label = "승격 기준은 넘었으나 기준 챔피언 이후의 변경과 충돌 → 승격하지 않음: " + "; ".join(d["blockers"])
                elif d["rule"].startswith("early"):
                    label = "조기 탈락 (avgR < -0.5)"
                elif d["rule"].startswith("trial over"):
                    label = f"시험 기간 종료 ({d['checkpoint']}일 판정점까지 승격 기준 미달)"
                else:
                    label = "챔피언 대비 열위 (HAC 일별 차이 t ≤ -1)" if d.get("retire_path") == "relative" else \
                        "절대 기준 열위 (HAC t_day ≤ -1)"
                await self.app.retire_challenger(slot, reason=f"{label}\n{describe_decision(d)}", decision=d)

    async def check_conflicts(self) -> list[ChallengerSlot]:
        """After a champion change (promotion, kill switch): retire every challenger whose change can no longer be
        applied onto the new champion (promotion_blockers), with the reason; it could never be promoted, and its slot
        is freed for a candidate built on the new champion. Returns the retired slots."""
        out = []
        for slot in list(self.app.challengers):
            blockers = self.promotion_blockers(slot)
            if not blockers:
                continue
            d = {"kind": "rebase", "verdict": "retire", "rule": "change conflicts with the current champion: "
                 + "; ".join(blockers), "blockers": blockers, "slot": slot.name, "version": slot.params.version,
                 "base": slot.base, "diff": self.slot_diff(slot), "champion_version": self.app.params.version,
                 "at": now_ms()}
            log.warning("retiring challenger %s (%s): its change conflicts with champion %s: %s", slot.name,
                        slot.params.version, self.app.params.version, "; ".join(blockers))
            await self.app.retire_challenger(slot, reason="기준 챔피언 이후의 변경과 충돌 (승격 불가): " + "; ".join(blockers),
                                             decision=d)
            out.append(slot)
        return out

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

    def _discard_pool(self, terminate: bool = False) -> None:
        pool, self.pool = self.pool, None
        if pool is None:
            return
        # the worker processes, read before shutdown() forgets them (a private attribute of the executor, guarded)
        procs = list((getattr(pool, "_processes", None) or {}).values()) if terminate else []
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            log.debug("research pool shutdown raised", exc_info=True)
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.monotonic() + 5.0
        for p in procs:
            try:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        """Stop the research / discovery worker for good. shutdown(wait=False) alone leaves a worker that is busy with
        a cycle running for minutes after the bot stopped (an orphan after SIGTERM, and interpreter exit then waits for
        it): it is terminated (killed if it does not exit within 5 s)."""
        self.stopped = True
        self._discard_pool(terminate=True)


CHAMPION_ENGINES = ("paper", "live")  # engines that trade the champion's parameters


# Kill-switch scale: the CUSUM runs on (R - expected) / sd, sd = the standard deviation of the alpha's R since its epoch,
# clamped to [KS_SD_FLOOR, KS_SD_CAP] R. Trade R is noisy (sd ~1-1.5R for stop-based alphas, more for trend followers
# with a low win rate) next to any real edge (~0.1R), so a CUSUM in raw R with a few R of alarm line fires on ordinary
# losing streaks: with k = 0.25R and h = 5R, a synthetic alpha earning +0.10R a trade at a 35% win rate tripped within
# 150 trades in 9 of 10 runs (tests/test_p4_killswitch.py). The floor keeps a run of identical losses (sd ~0) from
# dividing by ~0; the cap keeps one outsized winner from blinding it to a run of full losses.
KS_SD_FLOOR = 1.0
KS_SD_CAP = 2.0


class AlphaKillSwitch:
    """Per-alpha guard on live (paper or real) results: switches a losing alpha off at once (ROADMAP P4).

    For every engine (the paper champion, the live engine, each challenger) and every alpha it has switched on, the
    trades the engine opened with the alpha since its epoch are tracked: a one-sided lower CUSUM of
    (R - expected) / sd with slack KILL_SWITCH_K (sd units, see KS_SD_FLOOR), expected = the evidence-registry mean R if
    the alpha PASSED out of sample else 0, and the R sum of its last KILL_SWITCH_WINDOW trades (all of them while
    fewer). A CUSUM at or below -KILL_SWITCH_H (sd units), or a window sum below KILL_SWITCH_WINDOW_R, trips it:

    * a champion engine (paper / live) -> a new champion version with the alpha off, a Telegram alert, a log line.
      Switching off is the safe direction, so it needs no challenger; switching back on only happens through a
      promotion;
    * a challenger -> it is retired.

    The state is recomputed from the stored trades on every check, so a restart resumes exactly where the process
    stopped. The champion's epoch of an alpha (kv `killswitch.epoch`) is when it was last switched on or re-tuned in
    the champion (its live record before that judged other settings); a challenger's epoch is its start."""

    EPOCH_KEY = "killswitch.epoch"

    def __init__(self, app):
        self.app = app
        self.s = app.s
        self.store = app.store
        self.epochs: dict[str, int] = {k: int(v) for k, v in (self.store.get(self.EPOCH_KEY, {}) or {}).items()}
        self.state: dict[tuple[str, str], dict] = {}  # (engine, alpha) -> latest evaluation
        self.trips: list[dict] = []  # trips since start (newest last)
        self._acting: set[tuple[str, str]] = set()

    # --- epochs ------------------------------------------------------------------------------------
    def _save_epochs(self) -> None:
        self.store.set(self.EPOCH_KEY, self.epochs)

    def sync_champion(self, params: StrategyParams, default_epoch: int | None = None) -> None:
        """Every alpha the champion has on gets an epoch (the champion version's creation for one without), alphas
        it has off lose theirs."""
        changed = False
        for a, on in params.enabled.items():
            if on and a in params.alphas and a not in self.epochs:
                self.epochs[a] = int(default_epoch if default_epoch is not None else (params.created or now_ms()))
                changed = True
            elif (not on or a not in params.alphas) and a in self.epochs:
                del self.epochs[a]
                changed = True
        if changed:
            self._save_epochs()

    def on_new_champion(self, old: StrategyParams, new: StrategyParams, now: int) -> None:
        """A promotion: an alpha the new champion switches on, or runs with other parameters, starts a fresh record;
        alphas that stay as they were keep theirs."""
        for a, on in new.enabled.items():
            if on and a in new.alphas and (not old.enabled.get(a, False) or old.alphas.get(a) != new.alphas.get(a)):
                self.epochs[a] = int(now)
        self.sync_champion(new, now)
        self._save_epochs()

    # --- evaluation --------------------------------------------------------------------------------
    def _target(self, engine: str) -> dict[str, int] | None:
        """{alpha: since} for the alphas `engine` has switched on, None for an unknown engine."""
        if engine in CHAMPION_ENGINES:
            p = self.app.params
            return {a: int(self.epochs.get(a, p.created or 0)) for a, on in p.enabled.items() if on and a in p.alphas}
        slot = next((c for c in self.app.challengers if c.name == engine), None)
        if slot is None:
            return None
        return {a: int(slot.started) for a, on in slot.params.enabled.items() if on and a in slot.params.alphas}

    def evaluate(self, engine: str, alpha: str, since: int) -> dict:
        # trades the engine opened since the epoch (a position opened before it was decided under other settings)
        trades = sorted((t for t in self.store.load_trades(engine=engine, since=since, alpha=alpha)
                         if (t.get("entry_time") or 0) >= since), key=lambda t: t.get("exit_time") or 0)
        rs = [float(t.get("r_multiple") or 0.0) for t in trades]
        s = self.s
        exp = evidence.expected_r(alpha)
        sd = statistics.stdev(rs) if len(rs) >= 2 else 0.0
        scale = min(KS_SD_CAP, max(KS_SD_FLOOR, sd))
        cusum, low = cusum_lower(rs, exp, s.kill_switch_k, scale)
        win = rs[-s.kill_switch_window:] if s.kill_switch_window > 0 else rs
        wsum = sum(win)
        by_cusum = bool(rs) and cusum <= -s.kill_switch_h
        by_window = bool(rs) and wsum < s.kill_switch_window_r
        rule = " + ".join(x for x, hit in ((f"CUSUM <= -{s.kill_switch_h:g}sd", by_cusum),
                                           (f"last {len(win)} sum < {s.kill_switch_window_r:g}R", by_window)) if hit)
        st = {"engine": engine, "alpha": alpha, "since": int(since), "n": len(rs), "sum_r": sum(rs),
              "avg_r": sum(rs) / len(rs) if rs else 0.0, "expected": exp, "verdict": evidence.verdict(alpha),
              "sd": sd, "scale": scale, "cusum": cusum, "cusum_low": low, "k": s.kill_switch_k, "h": s.kill_switch_h,
              "window": s.kill_switch_window, "window_n": len(win), "window_sum": wsum, "window_r": s.kill_switch_window_r,
              "tripped": by_cusum or by_window, "rule": rule, "at": now_ms()}
        self.state[(engine, alpha)] = st
        return st

    async def check(self, engine: str, alpha: str | None = None) -> list[dict]:
        """Evaluate `engine` (one alpha or all it has on) and act on a trip. Returns the trips."""
        if not self.s.kill_switch or engine not in self.app.engines:
            return []
        since = self._target(engine)
        if not since:
            return []
        trips = []
        for a in ([alpha] if alpha else list(since)):
            if a not in since or (engine, a) in self._acting:
                continue  # not switched on in this engine (or already being acted on)
            st = self.evaluate(engine, a, since[a])
            if not st["tripped"]:
                continue
            self._acting.add((engine, a))
            try:
                await self._act(engine, a, st)
            finally:
                self._acting.discard((engine, a))
            trips.append(st)
            if engine not in CHAMPION_ENGINES:
                break  # the challenger is retired
        return trips

    async def check_all(self) -> list[dict]:
        """Startup: re-derive every engine's state from the stored trades (a trip the previous process did not get
        to act on, e.g. a crash right after the trade was stored, is acted on now)."""
        out: list[dict] = []
        for name in list(self.app.engines):
            out += await self.check(name)
        return out

    async def on_trade(self, payload: dict) -> None:
        """Bus handler for position_closed."""
        trade, engine = payload.get("trade"), payload.get("engine")
        alpha = getattr(trade, "alpha", None) if trade is not None else None
        if engine and alpha:
            await self.check(engine, alpha)

    async def _act(self, engine: str, alpha: str, st: dict) -> None:
        d = {"kind": "kill_switch", **_round(st)}
        self.trips.append(d)
        log.warning("kill switch: %s on %s tripped (%s): n=%d sum %+.2fR CUSUM %+.2fsd (sd %.2fR) last %d sum %+.2fR "
                    "expected %+.3fR", alpha, engine, st["rule"], st["n"], st["sum_r"], st["cusum"], st["scale"],
                    st["window_n"], st["window_sum"], st["expected"])
        if engine in CHAMPION_ENGINES:
            await self.app.kill_alpha(alpha, d)
        else:
            slot = next((c for c in self.app.challengers if c.name == engine), None)
            if slot is not None:
                await self.app.retire_challenger(slot, reason=f"킬 스위치\n{describe_kill(d)}", decision=d)

    def champion_state(self) -> list[dict]:
        """Latest state of every alpha the champion has on (paper engine, and live when attached)."""
        since = self._target("paper") or {}
        out = []
        for eng in CHAMPION_ENGINES:
            if eng not in self.app.engines:
                continue
            for a in since:
                st = self.state.get((eng, a))
                if st is None or st.get("since") != since[a]:
                    st = self.evaluate(eng, a, since[a])
                out.append(st)
        return out


def describe_kill(d: dict) -> str:
    """Korean rendering of a kill-switch evaluation (its inputs and thresholds). The CUSUM is in standard deviations
    of the alpha's R (σ, shown in R)."""
    return (f"{d.get('alpha')} ({d.get('engine')}): 거래 {d.get('n', 0)}건, 누적 {d.get('sum_r', 0):+.2f}R "
            f"(평균 {d.get('avg_r', 0):+.3f}R, 기대 {d.get('expected', 0):+.3f}R, {evidence.short(d.get('alpha', ''))})\n"
            f"CUSUM {d.get('cusum', 0):+.2f}σ (기준 -{d.get('h', 0):g}σ, k={d.get('k', 0):g}σ, "
            f"σ={d.get('scale', 1.0):.2f}R) · 최근 {d.get('window_n', 0)}건 합 {d.get('window_sum', 0):+.2f}R "
            f"(기준 {d.get('window_r', 0):g}R)")


def _watch_parent(ppid: int) -> None:
    while True:
        time.sleep(2.0)
        if os.getppid() != ppid:
            os._exit(0)  # the bot is gone (killed without a clean stop): never keep researching as an orphan


def _worker_init(ppid: int) -> None:
    """Research / discovery worker initializer: exit as soon as the parent process disappears."""
    threading.Thread(target=_watch_parent, args=(ppid,), daemon=True, name="parent-watch").start()


def _new_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_worker_init, initargs=(os.getpid(),))


def _discovery_worker(db_path: str, symbols: list[str], train: tuple[int, int], valid: tuple[int, int],
                      grids: str = "15m,1h") -> dict:
    """Runs in the research process: mine rules on every decision grid in ONE mine() call and return the validated ones
    (JSON-serialisable). One call makes the multiple-testing bar count every rule evaluated on every grid (N combined);
    mining the grids in separate calls validated each against its own, smaller N."""
    from heartless.learning import discovery as D

    out = {"tested": 0, "passed": [], "seconds": 0.0, "rows": {}, "t_bar": 0.0, "n_tests": 0, "tf": grids}
    try:
        res = D.mine(db_path, symbols, train, valid, grids, D.SearchConfig(beam=25, depth=3), workers=1)
    except Exception as e:  # noqa: BLE001
        out["errors"] = [f"{grids}: {e}"]
        return out
    out.update(tested=res["tested"], passed=list(res["passed"]), seconds=res["seconds"], rows=res["rows"],
               t_bar=res["t_bar"], n_tests=res["n_tests"], tf=res["tf"])
    out["passed"].sort(key=lambda r: -r.get("stats", {}).get("valid", {}).get("t", 0.0))
    return out
